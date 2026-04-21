import 'dotenv/config'; // Loads MUX_TOKEN_ID and MUX_TOKEN_SECRET

const ASSET_ID = 'H92Pdfgvd0200kcc5ZT7DwrwpCPI4lgcSTbb8Nogjfb02U';
const MUX_AUTH = Buffer.from(`${process.env.MUX_TOKEN_ID}:${process.env.MUX_TOKEN_SECRET}`).toString('base64');

async function fixAsset() {
  console.log(`🔍 Inspecting Asset: ${ASSET_ID}...`);

  // 1. Get Asset Details to find the Audio Track ID
  const assetResponse = await fetch(`https://api.mux.com/video/v1/assets/${ASSET_ID}`, {
    headers: { 'Authorization': `Basic ${MUX_AUTH}` }
  });
  const assetData = await assetResponse.json();

  if (!assetData.data) {
    console.error("❌ Could not find asset. Check your credentials.");
    return;
  }

  // Find the audio track
  const audioTrack = assetData.data.tracks.find(t => t.type === 'audio');

  if (!audioTrack) {
    console.error("❌ No audio track found in this video.");
    return;
  }

  console.log(`✅ Found Audio Track ID: ${audioTrack.id}`);
  console.log("🚀 Requesting auto-generated captions...");

  // 2. Request Captions for this specific track
  const generateResponse = await fetch(`https://api.mux.com/video/v1/assets/${ASSET_ID}/tracks/${audioTrack.id}/generate-subtitles`, {
    method: 'POST',
    headers: {
      'Authorization': `Basic ${MUX_AUTH}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      generated_subtitles: [{ language_code: 'en', name: 'English (Auto)' }]
    })
  });

  if (generateResponse.ok) {
    console.log("🎉 Success! Captions are now generating.");
    console.log("⏳ Please wait 2-5 minutes for Mux to process the text, then run your chapter script again.");
  } else {
    const err = await generateResponse.json();
    console.error("❌ Error:", JSON.stringify(err, null, 2));
  }
}

fixAsset();