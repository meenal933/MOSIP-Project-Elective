import 'dotenv/config';
import Mux from '@mux/mux-node';
import { GoogleGenerativeAI } from '@google/generative-ai';
import axios from 'axios';

// ---------------- CONFIGURATION ----------------
const ASSET_ID = 'H92Pdfgvd0200kcc5ZT7DwrwpCPI4lgcSTbb8Nogjfb02U';

const mux = new Mux({
  tokenId: process.env.MUX_TOKEN_ID,
  tokenSecret: process.env.MUX_TOKEN_SECRET
});

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GENERATIVE_AI_API_KEY);

// ---------------- MAIN LOGIC ----------------
async function main() {
  console.log(`🔍 Fetching Asset Details for: ${ASSET_ID}...`);

  try {
    // 1. Get Asset from Mux to find Playback ID and Track ID
    const asset = await mux.video.assets.retrieve(ASSET_ID);
    
    const playbackId = asset.playback_ids?.[0]?.id;
    const textTrack = asset.tracks?.find(t => t.type === 'text' && t.status === 'ready');

    if (!playbackId) throw new Error("No Playback ID found for this asset.");
    if (!textTrack) throw new Error("No READY text track found. Did you run fix_captions.js?");

    console.log(`   > Playback ID: ${playbackId}`);
    console.log(`   > Track ID: ${textTrack.id}`);

    // 2. Download the VTT Captions
    // Public URL format: https://stream.mux.com/{PLAYBACK_ID}/text/{TRACK_ID}.vtt
    const vttUrl = `https://stream.mux.com/${playbackId}/text/${textTrack.id}.vtt`;
    console.log(`📥 Downloading captions from: ${vttUrl}`);
    
    const vttResponse = await axios.get(vttUrl);
    const vttContent = vttResponse.data;

    // 3. Send to Gemini
    console.log("🤖 Sending transcript to Gemini 1.5 Flash...");
    const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

    const prompt = `
      You are a video editor. Here is a WebVTT caption file for a technical tutorial about MOSIP.
      
      TASK:
      Generate a JSON list of chapters for this video.
      - Each chapter must have a 'startTime' (in seconds, as an integer) and a 'title'.
      - Titles should be descriptive but concise (under 50 chars).
      - Ignore filler talk; focus on technical steps.
      
      OUTPUT FORMAT (JSON ONLY):
      [
        { "startTime": 0, "title": "Introduction" },
        { "startTime": 150, "title": "Next Topic..." }
      ]

      TRANSCRIPT:
      ${vttContent.substring(0, 30000)} // Truncating just in case, though 1.5 Flash handles huge context.
    `;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    let text = response.text();

    // Cleanup: Remove any Markdown code blocks if Gemini adds them
    text = text.replace(/```json/g, '').replace(/```/g, '').trim();

    console.log("\n✅ Generated Chapters JSON:");
    console.log(text);

  } catch (error) {
    console.error("❌ Error:", error.message);
    if (error.response) console.error("   > API Details:", error.response.data);
  }
}

main();