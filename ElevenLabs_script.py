import os
import subprocess
import re
import requests
from dotenv import load_dotenv
from groq import Groq 

# Load API Keys
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY) # <-- NEW

# Configuration
INPUT_VIDEO = "input_video.mp4" 
OUTPUT_DIR = "experiments"
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  
TTS_MODEL = "eleven_multilingual_v2" 
STT_MODEL = "scribe_v1"
GROQ_MODEL = "llama-3.1-8b-instant"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "xi-api-key": ELEVENLABS_API_KEY
}

def extract_audio_for_stt(video_path):
    """Extracts audio from the video for STT processing."""
    base_name = os.path.basename(video_path).replace(".mp4", "")
    audio_path = os.path.join(OUTPUT_DIR, f"{base_name}_temp.mp3")
    
    cmd = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def transcribe_audio(audio_file):
    """Uses ElevenLabs Scribe API for STT."""
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    
    with open(audio_file, "rb") as f:
        files = {"file": f}
        data = {"model_id": STT_MODEL}
        response = requests.post(url, headers=HEADERS, files=files, data=data)
    
    if response.status_code != 200:
        raise Exception(f"STT Error: {response.text}")
        
    return response.json().get("text", "")

def remove_fillers_and_clean(text):
    """
    Uses Groq (LLaMA 3) to clean up the transcript.
    Includes a fallback to regex just in case the API call fails.
    """
    # We use a strict prompt to ensure it doesn't hallucinate new text 
    # or change the meaning, which would ruin the video pacing.
    system_prompt = (
        "You are an expert audio transcript editor. Your task is to clean up the provided "
        "transcript by removing filler words (um, uh, like, you know), stuttering, and correcting "
        "minor grammatical errors. "
        "CRITICAL RULES: Do NOT rewrite the text, change the core meaning, or add new information. "
        "Output ONLY the cleaned text, without any introductory or concluding remarks."
    )

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript: {text}"}
            ],
            model=GROQ_MODEL,
            temperature=0.1, # Keep temperature very low so it doesn't get "creative"
        )
        
        clean_text = chat_completion.choices[0].message.content.strip()
        return clean_text
        
    except Exception as e:
        print(f"   [!] Groq LLM Error: {e}. Falling back to Regex.")
        # Regex Fallback so your pipeline doesn't crash entirely if Groq times out
        clean_text = re.sub(r'\b(um|uh|like|you know|sort of)\b', '', text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', clean_text).strip()

def generate_audio(text, output_audio_path):
    """Uses ElevenLabs TTS API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    
    payload = {
        "text": text,
        "model_id": TTS_MODEL,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    response = requests.post(url, json=payload, headers=HEADERS)
    
    if response.status_code != 200:
        raise Exception(f"TTS Error: {response.text}")
        
    with open(output_audio_path, "wb") as f:
        f.write(response.content)
        
    return output_audio_path

def merge_audio_video(video_path, new_audio_path, final_output_path):
    """Replaces original audio with new AI audio."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", new_audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"   [✓] Final video ready: {final_output_path}")

def main():
    if not ELEVENLABS_API_KEY:
        print("ERROR: ELEVENLABS_API_KEY not found in .env file")
        return
        
    if not os.path.exists(INPUT_VIDEO):
        print(f"ERROR: Could not find {INPUT_VIDEO}. Please place your 2-min video in the directory.")
        return

    print("--- Starting Single Video Test Pipeline ---")
    base_name = os.path.basename(INPUT_VIDEO).replace(".mp4", "")
    
    # 1. Extract Audio & Transcribe (Scribe STT)
    print("   ... Extracting audio & Transcribing (Scribe)")
    temp_audio = extract_audio_for_stt(INPUT_VIDEO)
    raw_text = transcribe_audio(temp_audio)
    
    # Optional: Print a snippet of the transcription to verify it worked
    print(f"   [Transcription Snippet]: {raw_text[:100]}...")
    
    # 2. Clean Text (Remove Fillers)
    print("   ... Cleaning text (Regex)")
    clean_text = remove_fillers_and_clean(raw_text)
    
    # 3. Generate New Audio (TTS)
    print(f"   ... Generating Synthetic Voice ({TTS_MODEL})")
    new_audio_output = os.path.join(OUTPUT_DIR, f"{base_name}_dubbed.mp3")
    generate_audio(clean_text, new_audio_output)
    
    # 4. Merge
    print("   ... Merging Final Video")
    final_output = os.path.join(OUTPUT_DIR, f"{base_name}_final.mp4")
    merge_audio_video(INPUT_VIDEO, new_audio_output, final_output)
    
    # Cleanup temp files
    print("   ... Cleaning up temporary files")
    for temp_file in [temp_audio, new_audio_output]:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(f"\n--- Pipeline Complete! Check the '{OUTPUT_DIR}' folder. ---")

if __name__ == "__main__":
    main()