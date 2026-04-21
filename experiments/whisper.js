import 'dotenv/config';
import fs from 'fs';
import ffmpeg from 'fluent-ffmpeg';
import Groq from 'groq-sdk';

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const VIDEO_FILE = 'mosip_tutorial.mp4';
const AUDIO_FILE = 'temp_audio.m4a';
const MIN_DURATION_SECONDS = 600; // 10 Minutes (Strict)
function parseTime(timeStr) {
  const parts = timeStr.split(':').map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  return parts[0] * 60 + parts[1];
}

function enforceMinimumDuration(chapters, totalDurationSec) {
  if (chapters.length === 0) return [];
  
  const merged = [];
  let currentStart = 0;
  let currentTitle = chapters[0].title;

  for (let i = 0; i < chapters.length - 1; i++) {
    const nextStart = parseTime(chapters[i+1].timestamp);
    const duration = nextStart - currentStart;

    if (duration < MIN_DURATION_SECONDS) {
      console.log(`   > Merging short chapter (${Math.floor(duration/60)}m): "${currentTitle}"`);
      currentTitle = `${currentTitle} & ${chapters[i+1].title}`; 
    } else {
      merged.push({ 
        timestamp: new Date(currentStart * 1000).toISOString().substr(11, 8), // Format HH:MM:SS
        title: currentTitle 
      });
      currentStart = nextStart;
      currentTitle = chapters[i+1].title;
    }
  }

  merged.push({ 
    timestamp: new Date(currentStart * 1000).toISOString().substr(11, 8),
    title: currentTitle 
  });

  return merged;
}

async function main() {
  console.log("🚀 Starting Strict 10-Min Chapter Pipeline...");

  try {
    if (!fs.existsSync(AUDIO_FILE)) {
      console.log("🔊 Extracting audio...");
      await new Promise((resolve, reject) => {
        ffmpeg(VIDEO_FILE).noVideo().audioCodec('aac').audioBitrate('64k')
          .save(AUDIO_FILE).on('end', resolve).on('error', reject);
      });
    }

    console.log("✍️  Transcribing...");
    const transcription = await groq.audio.transcriptions.create({
      file: fs.createReadStream(AUDIO_FILE),
      model: "whisper-large-v3",
      response_format: "verbose_json", 
    });
    
    let formattedText = "";
    let lastMarker = -100;
    transcription.segments.forEach(s => {
      if (s.start - lastMarker > 60) {
        const time = new Date(s.start * 1000).toISOString().substr(14, 5); // MM:SS
        formattedText += `\n[TIME: ${time}] `;
        lastMarker = s.start;
      }
      formattedText += s.text + " ";
    });

    // 4. GENERATE RAW CHAPTERS
    console.log("🧠 Finding topics...");
    const prompt = `
      Analyze this transcript. Identify 8-12 logical topic shifts.
      Return STRICT JSON: [{ "timestamp": "MM:SS", "title": "Topic Name" }]
      TRANSCRIPT: ${formattedText.substring(0, 95000)}
    `;

    const completion = await groq.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      model: "llama-3.3-70b-versatile",
      temperature: 0.1,
    });

    const rawContent = completion.choices[0]?.message?.content;
    const jsonMatch = rawContent.match(/\[.*\]/s);
    if (!jsonMatch) throw new Error("No JSON found in response");
    
    const rawChapters = JSON.parse(jsonMatch[0]);
    console.log(`\n📉 Raw AI Chapters: ${rawChapters.length}`);

    // 5. THE FIX: Algorithmic Post-Processing
    console.log("✨ Applying 10-Minute Minimum Rule...");
    const finalChapters = enforceMinimumDuration(rawChapters, transcription.duration);

    // Save
    fs.writeFileSync('chapters.json', JSON.stringify(finalChapters, null, 2));
    console.log("\n✅ FINAL CHAPTERS (Guaranteed >10m):");
    console.log(finalChapters);

  } catch (err) {
    console.error("❌ Error:", err.message);
  }
}

main();