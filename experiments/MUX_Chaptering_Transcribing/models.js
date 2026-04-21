import 'dotenv/config';
import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GENERATIVE_AI_API_KEY);

async function listModels() {
  console.log("🔍 Checking available Gemini models for your API key...");
  try {
    const models = await genAI.getGenerativeModel({ model: "gemini-pro" }).apiKey; 
    // The SDK doesn't have a simple "list" method in all versions, 
    // so we use a specific fetch to the API endpoint to be sure.
    
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${process.env.GOOGLE_GENERATIVE_AI_API_KEY}`);
    const data = await response.json();

    if (data.models) {
        console.log("\n✅ AVAILABLE MODELS:");
        data.models.forEach(m => {
            if (m.name.includes("gemini")) {
                console.log(` - ${m.name.replace("models/", "")}`);
            }
        });
    } else {
        console.log("❌ Error listing models:", data);
    }
  } catch (err) {
    console.error("❌ Network Error:", err.message);
  }
}

listModels();