import 'dotenv/config';
import ffmpeg from 'fluent-ffmpeg';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { GoogleAIFileManager } from '@google/generative-ai/server';
import fs from 'fs';

const VIDEO_FILENAME = 'mosip_tutorial.mp4';
const OUTPUT_JSON_FILE = 'chapters.json';
const TEMP_AUDIO_FILE = 'temp_audio_extract.mp3';
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GENERATIVE_AI_API_KEY);
const fileManager = new GoogleAIFileManager(process.env.GOOGLE_GENERATIVE_AI_API_KEY);

async function main() {
  console.log("🚀 Starting Video-to-JSON Pipeline...");

  try {
    console.log(`🔊 Extracting audio from ${VIDEO_FILENAME}...`);
    await new Promise((resolve, reject) => {
      ffmpeg(VIDEO_FILENAME)
        .noVideo()
        .audioCodec('libmp3lame')
        .audioBitrate(128)
        .save(TEMP_AUDIO_FILE)
        .on('end', resolve)
        .on('error', (err) => reject(new Error(`FFmpeg Error: ${err.message}`)));
    });
    console.log("   > Audio extracted successfully.");

    console.log("☁️  Uploading audio to Gemini storage...");
    const uploadResult = await fileManager.uploadFile(TEMP_AUDIO_FILE, {
      mimeType: "audio/mp3",
      displayName: "Automated Extraction",
    });

    let file = await fileManager.getFile(uploadResult.file.name);
    process.stdout.write("   > Processing remote file");
    while (file.state === "PROCESSING") {
      process.stdout.write(".");
      await new Promise((r) => setTimeout(r, 2000));
      file = await fileManager.getFile(uploadResult.file.name);
    }
    console.log(`\n   > Ready. URI: ${file.uri}`);

    console.log("🧠 Analyzing audio for chapters...");
    const model = genAI.getGenerativeModel({ 
      model: "gemini-2.0-flash-lite",
      generationConfig: { responseMimeType: "application/json" }
    });

    const prompt = `
      You are an automated video processing engine.
      
      Task:
      1. Listen to the audio heavily.
      2. Identify the logical sections/chapters of the content.
      3. Extract the exact start timestamp for each section.
      4. Create a concise, technical title for each section.
      
      Output Schema (Strict JSON):
      [
        {
          "timestamp": "MM:SS",
          "seconds": 0,
          "title": "Topic Name"
        }
      ]
      
      Requirements:
      - 'timestamp': Format as HH:MM:SS or MM:SS depending on duration.
      - 'seconds': The raw start time in seconds (integer).
      - 'title': No fluff. Technical and direct.
    `;

    const result = await model.generateContent([
      { fileData: { mimeType: file.mimeType, fileUri: file.uri } },
      { text: prompt }
    ]);

    // STEP 4: Save to JSON File
    const jsonString = result.response.text();
    fs.writeFileSync(OUTPUT_JSON_FILE, jsonString);
    
    console.log(`\n✅ DONE! Data stored in: ${OUTPUT_JSON_FILE}`);
    console.log("------------------------------------------------");
    console.log(jsonString.substring(0, 200) + "..."); // Preview

    // Clean up temp file
    if (fs.existsSync(TEMP_AUDIO_FILE)) fs.unlinkSync(TEMP_AUDIO_FILE);

  } catch (error) {
    console.error("\n❌ CRITICAL ERROR:", error.message);
  }
}

main();