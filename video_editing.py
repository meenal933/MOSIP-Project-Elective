import json
import os
import subprocess
import re
import sys
import argparse
import time
from dotenv import load_dotenv
from deepgram import DeepgramClient
import google.generativeai as genai
from groq import Groq

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_GENERATIVE_AI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

INPUT_VIDEO = "mosip_tutorial.mp4"
CHAPTERS_FILE = "chapters.json"
OUTPUT_DIR = "processed_chapters"
DEEPGRAM_VOICE = "aura-2-odysseus-en"
LLM_PROVIDER = "groq" 

os.makedirs(OUTPUT_DIR, exist_ok=True)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

def parse_timestamp(timestamp_str):
    """Converts MM:SS or HH:MM:SS to seconds."""
    parts = list(map(int, timestamp_str.split(':')))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def split_video(start_time, end_time, index, title):
    """Splits the video into a chapter using FFmpeg."""
    clean_title = re.sub(r'[^a-zA-Z0-9]', '_', title)
    output_filename = os.path.join(OUTPUT_DIR, f"chap_{index}_{clean_title}.mp4")
    
    cmd = ["ffmpeg", "-y", "-i", INPUT_VIDEO, "-ss", str(start_time)]
    
    if end_time:
        duration = end_time - start_time
        cmd.extend(["-t", str(duration)])
    
    cmd.extend(["-c:v", "libx264", "-c:a", "aac", output_filename])
    
    print(f"   [CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"   [!] FFmpeg Error: {result.stderr.decode()}")
        return None
        
    print(f"   [✓] Split video created: {output_filename}")
    return output_filename

def transcribe_video(client, video_file):
    """Transcribes audio using Deepgram."""
    if not os.path.exists(video_file):
        print(f"   [!] File not found: {video_file}")
        return ""

    with open(video_file, "rb") as file:
        buffer_data = file.read()

    try:
        response = client.listen.v1.media.transcribe_file(
            request=buffer_data,
            model="nova-2",
            smart_format=True,
            punctuate=True,
            language="en"
        )
        
        if response and response.results:
            return response.results.channels[0].alternatives[0].transcript
        return ""
    except Exception as e:
        print(f"   [!] Deepgram Transcription Error: {e}")
        return ""

def refine_transcript_with_ai(transcript, chapter_title):
    """Uses LLM to refine the entire transcript - clean it without summarizing."""
    
    prompt = f"""You are a professional script editor. Your task is to refine a video transcript to make it cleaner and more professional.

## Chapter: {chapter_title}

## Original Transcript:
{transcript}

## Rules:
1. Keep the SAME content and meaning - do not summarize or remove information
2. Remove filler words like "right", "you know", "so", "like", "um", "uh" where they don't add meaning
3. Fix grammar and awkward phrasing
4. Make it sound natural when read aloud by a professional narrator
5. Keep the same approximate length - just make it cleaner
6. Keep technical terms and proper nouns unchanged
7. Maintain the conversational but professional tone

## Output:
Return ONLY the refined transcript text, nothing else. No JSON, no explanations."""

    result = None
    
    # Try Groq first
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        try:
            print(f"   --> Using Groq (llama-3.3-70b-versatile)")
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=8000
            )
            result = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"   [!] Groq Error: {e}")
    
    # Fallback to Gemini
    if result is None and GEMINI_API_KEY:
        try:
            print(f"   --> Using Gemini (gemini-2.0-flash)")
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            result = response.text.strip()
        except Exception as e:
            print(f"   [!] Gemini Error: {e}")
    
    return result if result else transcript

def split_text_for_tts(text, max_chars=1900):
    """Split text into chunks respecting TTS character limit."""
    if len(text) <= max_chars:
        return [text]
    
    sentences = text.replace('? ', '?|').replace('. ', '.|').replace('! ', '!|').split('|')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_audio_chunk(client, text, output_path):
    """Generate audio for a single text chunk."""
    response = client.speak.v1.audio.generate(
        text=text,
        model=DEEPGRAM_VOICE
    )
    
    with open(output_path, "wb") as audio_file:
        if hasattr(response, 'stream'):
            audio_file.write(response.stream.getvalue())
        elif hasattr(response, '__iter__'):
            for chunk in response:
                if isinstance(chunk, bytes):
                    audio_file.write(chunk)
                elif hasattr(chunk, 'data'):
                    audio_file.write(chunk.data)
        else:
            audio_file.write(response)
    return output_path

def generate_audio(client, text, output_audio_path):
    """Generate TTS audio for the refined transcript."""
    if not text:
        print("   [!] No text to generate audio from.")
        return None

    base_path = output_audio_path.replace(".mp3", "")
    
    try:
        chunks = split_text_for_tts(text, max_chars=1900)
        print(f"   --> Generating audio for {len(chunks)} chunk(s)")
        
        if len(chunks) == 1:
            generate_audio_chunk(client, chunks[0], output_audio_path)
            return output_audio_path
        
        # Generate audio for each chunk
        chunk_files = []
        for i, chunk_text in enumerate(chunks):
            chunk_path = f"{base_path}_chunk_{i}.mp3"
            generate_audio_chunk(client, chunk_text, chunk_path)
            chunk_files.append(chunk_path)
        
        # Concatenate all chunks
        concat_list_path = f"{base_path}_concat.txt"
        with open(concat_list_path, "w") as f:
            for chunk_file in chunk_files:
                f.write(f"file '{os.path.abspath(chunk_file)}'\n")
        
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c:a", "libmp3lame", output_audio_path
        ]
        subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        if os.path.exists(concat_list_path):
            os.remove(concat_list_path)
        
        return output_audio_path

    except Exception as e:
        print(f"   [!] TTS Error: {e}")
        return None

def get_duration(file_path):
    """Get duration of audio/video file using FFprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return None

def speed_up_video_to_match_audio(video_path, audio_path, output_path):
    """Speed up video to match audio duration, then combine them."""
    
    video_duration = get_duration(video_path)
    audio_duration = get_duration(audio_path)
    
    if not video_duration or not audio_duration:
        print("   [!] Could not determine durations")
        return None
    
    speed_factor = video_duration / audio_duration
    print(f"   --> Video: {video_duration:.1f}s, Audio: {audio_duration:.1f}s")
    print(f"   --> Speed factor: {speed_factor:.2f}x (video will be {100/speed_factor:.0f}% of original)")
    
    if speed_factor < 0.5 or speed_factor > 4.0:
        print(f"   [!] Speed factor {speed_factor:.2f} is too extreme. Limiting to safe range.")
        speed_factor = max(0.5, min(speed_factor, 4.0))
    
    # Speed up video using setpts filter
    # setpts=PTS/speed_factor makes video faster
    # For example, setpts=PTS/1.5 makes video 1.5x faster
    
    pts_factor = 1 / speed_factor  # Inverse for setpts
    
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-filter:v", f"setpts={pts_factor:.4f}*PTS",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    
    print(f"   [CMD] ffmpeg ... -filter:v setpts={pts_factor:.4f}*PTS ...")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"   [!] FFmpeg Error: {result.stderr.decode()[:200]}")
        return None
    
    # Verify output
    final_duration = get_duration(output_path)
    print(f"   --> Final video duration: {final_duration:.1f}s (was {video_duration:.1f}s)")
    
    return output_path

def process_chapter(client, chapter_video, chapter_title, video_duration):
    """Process a single chapter: transcribe, refine, generate audio, speed up video."""
    
    # 1. Transcribe
    print("   ... Transcribing (Nova-2)")
    transcript = transcribe_video(client, chapter_video)
    
    if not transcript:
        print("   [!] No transcript generated.")
        return None
         
    print(f"   --> Transcript: {len(transcript)} chars")
    
    # Save original
    with open(chapter_video.replace(".mp4", "_original.txt"), "w", encoding="utf-8") as tf:
        tf.write(transcript)

    # 2. Refine with AI
    print("   ... Refining with AI")
    refined_transcript = refine_transcript_with_ai(transcript, chapter_title)
    
    # Save refined
    with open(chapter_video.replace(".mp4", "_refined.txt"), "w", encoding="utf-8") as tf:
        tf.write(refined_transcript)
    
    print(f"   --> Refined: {len(refined_transcript)} chars")

    # 3. Generate TTS audio
    print(f"   ... Generating Voice ({DEEPGRAM_VOICE})")
    audio_output = chapter_video.replace(".mp4", "_audio.mp3")
    
    result = generate_audio(client, refined_transcript, audio_output)
    
    if not result or not os.path.exists(audio_output):
        print("   [!] Audio file not created.")
        return None

    # 4. Speed up video to match audio and combine
    print("   ... Speeding up video to match audio")
    final_output = chapter_video.replace(".mp4", "_final.mp4")
    result = speed_up_video_to_match_audio(chapter_video, audio_output, final_output)
    
    if result:
        print(f"   [✓] Final video ready: {final_output}")
        return final_output
    
    return None

def main():
    parser = argparse.ArgumentParser(description='AI Video Repurposing Pipeline - Speed Up Edition')
    parser.add_argument('--chapter', type=int, help='Process only specific chapter (1-indexed)')
    args = parser.parse_args()
    
    if not DEEPGRAM_API_KEY:
        print("ERROR: DEEPGRAM_API_KEY not found in .env file")
        return
    
    if not GROQ_API_KEY and not GEMINI_API_KEY:
        print("ERROR: No LLM API key found (GROQ_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY)")
        return
        
    try:
        client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Deepgram Client: {e}")
        return

    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: Input video '{INPUT_VIDEO}' not found.")
        return

    if not os.path.exists(CHAPTERS_FILE):
        print(f"ERROR: {CHAPTERS_FILE} not found.")
        return

    with open(CHAPTERS_FILE, 'r') as f:
        chapters = json.load(f)

    print(f"=== AI Video Repurposing Pipeline (Speed Up Edition) ===")
    print(f"Input Video: {INPUT_VIDEO}")
    print(f"Chapters: {len(chapters)}")
    print(f"LLM Provider: {LLM_PROVIDER}")
    print()

    chapters_to_process = range(len(chapters))
    if args.chapter:
        chapters_to_process = [args.chapter - 1]

    for i in chapters_to_process:
        if i < 0 or i >= len(chapters):
            print(f"Invalid chapter number: {i + 1}")
            continue
            
        chapter = chapters[i]
        print(f"\n{'='*60}")
        print(f"Processing Chapter {i+1}: {chapter['title']}")
        print(f"{'='*60}")
        
        start = parse_timestamp(chapter['timestamp'])
        if i < len(chapters) - 1:
            end = parse_timestamp(chapters[i+1]['timestamp'])
        else:
            end = None 
        
        # 1. Split video
        print(f"   ... Splitting video ({start}s -> {end if end else 'End'}s)")
        chapter_video = split_video(start, end, i+1, chapter['title'])
        
        if not chapter_video or not os.path.exists(chapter_video):
            print("   [!] Splitting failed, skipping.")
            continue
        
        # Get video duration
        video_duration = get_duration(chapter_video)
        print(f"   --> Duration: {video_duration:.2f}s" if video_duration else "   --> Duration: Unknown")
        
        # 2. Process chapter
        result = process_chapter(client, chapter_video, chapter['title'], video_duration)
        
        if result:
            print(f"   [✓] Chapter {i+1} complete!")

    print("\n=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()
