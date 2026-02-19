from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
from google import genai
from google.genai import types

# 1. Initialize FastAPI
app = FastAPI()

# 2. Add CORS so your workmate's web app can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize Gemini Client
# Remember to add GEMINI_API_KEY in Vercel Dashboard Settings!
client = genai.Client()

class ComplaintRequest(BaseModel):
    text: str

# Home route to confirm it is live
@app.get("/")
async def home():
    return {"status": "Barangay AI is live on Vercel", "test": "Go to /docs"}

# The main classification endpoint
@app.post("/api/classify")
async def analyze_complaint(request: ComplaintRequest):
    prompt = f"""
    You are an AI dispatcher for a Philippine Barangay.
    Analyze the following complaint: "{request.text}"
    Categorize it into one of these Incident Types: [Theft & Robbery, Physical Injury, Fire & Disaster, Medical Emergency, VAWC, Public Disturbance, General Incident]
    And one of these Urgency Levels: [Critical, High, Medium, Low]
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "incident_type": {"type": "STRING"},
                        "urgency_level": {"type": "STRING"}
                    },
                    "required": ["incident_type", "urgency_level"]
                },
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}
