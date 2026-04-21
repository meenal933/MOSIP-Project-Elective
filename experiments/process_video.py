import json
import os
import subprocess
import re
import sys
from dotenv import load_dotenv
from deepgram import DeepgramClient

# Load API Key
load_dotenv()
API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Configuration
INPUT_VIDEO = "mosip_tutorial.mp4"
CHAPTERS_FILE = "chapters.json"
OUTPUT_DIR = "processed_chapters"
DEEPGRAM_VOICE = "aura-asteria-en" #model=aura-2-odysseus-en

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def transcribe_with_timestamps(client, video_file):
    """Transcribes audio with word-level timestamps using Deepgram."""
    if not os.path.exists(video_file):
        print(f"   [!] File not found: {video_file}")
        return None, ""

    with open(video_file, "rb") as file:
        buffer_data = file.read()

    try:
        # Request utterances for sentence-level timing
        response = client.listen.v1.media.transcribe_file(
            request=buffer_data,
            model="nova-2",
            smart_format=True,
            punctuate=True,
            language="en",
            utterances=True  # Get utterance-level timestamps
        )
        
        if response and response.results:
            transcript = response.results.channels[0].alternatives[0].transcript
            
            # Get utterances (sentences with timestamps)
            utterances = []
            if hasattr(response.results, 'utterances') and response.results.utterances:
                for utt in response.results.utterances:
                    utterances.append({
                        'text': utt.transcript,
                        'start': utt.start,
                        'end': utt.end
                    })
            else:
                # Fallback: use word-level timestamps to build sentences
                words = response.results.channels[0].alternatives[0].words
                if words:
                    current_sentence = []
                    sentence_start = words[0].start if words else 0
                    
                    for word in words:
                        current_sentence.append(word.word)
                        # Check if word ends with sentence punctuation
                        if word.word.endswith('.') or word.word.endswith('?') or word.word.endswith('!'):
                            utterances.append({
                                'text': ' '.join(current_sentence),
                                'start': sentence_start,
                                'end': word.end
                            })
                            current_sentence = []
                            sentence_start = word.end
                    
                    # Add remaining words as last utterance
                    if current_sentence:
                        utterances.append({
                            'text': ' '.join(current_sentence),
                            'start': sentence_start,
                            'end': words[-1].end
                        })
            
            return utterances, transcript
        return None, ""
    except Exception as e:
        print(f"   [!] Deepgram Transcription Error: {e}")
        return None, ""

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

def get_audio_duration(audio_path):
    """Get duration of audio file using FFprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return 0

def generate_silence(duration, output_path):
    """Generate silence audio of specified duration."""
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=24000:cl=mono",
        "-t", str(duration),
        "-c:a", "libmp3lame", output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

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

def generate_timed_audio(client, utterances, output_audio_path, video_duration):
    """Generate audio with timing to match original video."""
    if not utterances:
        print("   [!] No utterances to generate audio from.")
        return None

    base_path = output_audio_path.replace(".mp3", "")
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    audio_segments = []
    current_time = 0
    
    try:
        for i, utt in enumerate(utterances):
            target_start = utt['start']
            target_end = utt['end']
            text = utt['text']
            
            # Add silence if there's a gap before this utterance
            if target_start > current_time:
                silence_duration = target_start - current_time
                silence_path = f"{base_path}_silence_{i}.mp3"
                generate_silence(silence_duration, silence_path)
                audio_segments.append(silence_path)
                current_time = target_start
            
            # Split text if too long for TTS
            text_chunks = split_text_for_tts(text)
            
            for j, chunk_text in enumerate(text_chunks):
                chunk_path = f"{base_path}_utt_{i}_{j}.mp3"
                generate_audio_chunk(client, chunk_text, chunk_path)
                audio_segments.append(chunk_path)
            
            # Update current time based on target end
            current_time = target_end
        
        # Add trailing silence if video is longer
        if video_duration and current_time < video_duration:
            trailing_silence = f"{base_path}_trailing.mp3"
            generate_silence(video_duration - current_time, trailing_silence)
            audio_segments.append(trailing_silence)
        
        # Concatenate all segments
        if len(audio_segments) == 1:
            os.rename(audio_segments[0], output_audio_path)
        else:
            concat_list_path = f"{base_path}_concat.txt"
            with open(concat_list_path, "w") as f:
                for seg_file in audio_segments:
                    f.write(f"file '{os.path.abspath(seg_file)}'\n")
            
            concat_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-c:a", "libmp3lame", output_audio_path
            ]
            subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Cleanup
            for seg_file in audio_segments:
                if os.path.exists(seg_file):
                    os.remove(seg_file)
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)
        
        print(f"   [✓] Generated timed audio with {len(utterances)} utterances")
        return output_audio_path

    except Exception as e:
        print(f"   [!] Deepgram TTS Error: {e}")
        return None

def get_video_duration(video_path):
    """Get duration of video file using FFprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return None

def merge_audio_video(video_path, new_audio_path, final_output_path):
    """Replaces the audio in the video with the new audio."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", new_audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"   [✓] Final video ready: {final_output_path}")

def main():
    if not API_KEY:
        print("ERROR: DEEPGRAM_API_KEY not found in .env file")
        return
        
    try:
        client = DeepgramClient(api_key=API_KEY)
    except Exception as e:
        print(f"Failed to initialize Deepgram Client: {e}")
        return

    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: Input video '{INPUT_VIDEO}' not found.")
        return

    if not os.path.exists(CHAPTERS_FILE):
        print(f"ERROR: {CHAPTERS_FILE} not found. Please create it.")
        return

    with open(CHAPTERS_FILE, 'r') as f:
        chapters = json.load(f)

    print(f"--- Starting Pipeline for {len(chapters)} Chapters ---")
    print(f"Input Video: {INPUT_VIDEO}")

    for i, chapter in enumerate(chapters):
        print(f"\nProcessing Chapter {i+1}: {chapter['title']}")
        
        start = parse_timestamp(chapter['timestamp'])
        if i < len(chapters) - 1:
            end = parse_timestamp(chapters[i+1]['timestamp'])
        else:
            end = None 
        
        # 1. Split
        print(f"   ... Splitting video ({start}s -> {end if end else 'End'}s)")
        chapter_video = split_video(start, end, i+1, chapter['title'])
        
        if not chapter_video or not os.path.exists(chapter_video):
            print("   [!] Splitting failed, skipping.")
            continue
        
        # Get video duration for timing
        video_duration = get_video_duration(chapter_video)
        print(f"   --> Video duration: {video_duration:.2f}s" if video_duration else "   --> Could not get video duration")
        
        # 2. Transcribe with timestamps
        print("   ... Transcribing with timestamps (Nova-2)")
        utterances, transcript = transcribe_with_timestamps(client, chapter_video)
        
        if not transcript:
            print("   [!] No transcript generated, skipping TTS.")
            continue
             
        print(f"   --> Transcript length: {len(transcript)} chars, {len(utterances) if utterances else 0} utterances")
        
        # Save transcript
        with open(chapter_video.replace(".mp4", ".txt"), "w", encoding="utf-8") as tf:
            tf.write(transcript)
        
        # Save utterances with timestamps for debugging
        with open(chapter_video.replace(".mp4", "_timestamps.json"), "w", encoding="utf-8") as tf:
            json.dump(utterances, tf, indent=2)

        # 3. Generate timed audio
        print(f"   ... Generating timed Voice ({DEEPGRAM_VOICE})")
        audio_output = chapter_video.replace(".mp4", "_audio.mp3")
        
        result = generate_timed_audio(client, utterances, audio_output, video_duration)
        
        if not result or not os.path.exists(audio_output):
            print("   [!] Audio file not created, skipping merge.")
            continue

        # 4. Merge
        print("   ... Merging")
        final_output = chapter_video.replace(".mp4", "_final.mp4")
        merge_audio_video(chapter_video, audio_output, final_output)

    print("\n--- Pipeline Complete! ---")

if __name__ == "__main__":
    main()