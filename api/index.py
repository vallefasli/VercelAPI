from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import os

import json

from google import genai

from google.genai import types
 
app = FastAPI()
 
app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)
 
api_key = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
 
class ComplaintRequest(BaseModel):

    text: str
 
@app.get("/")

async def home():

    return {

        "status": "B.E.T.I.N.A. AI API is live on Vercel!",

        "docs": "/docs"

    }
 
@app.post("/api/classify")

async def analyze_complaint(request: ComplaintRequest):

    prompt = f"""

    You are an AI dispatcher for a Philippine Barangay.

    Analyze the following complaint: "{request.text}"

    Categorize it strictly into one of these Incident Types:

    [Theft & Robbery, Physical Injury, Fire & Disaster, Medical Emergency, VAWC, Public Disturbance, General Incident]

    And categorize it strictly into one of these Urgency Levels:

    [Critical, High, Medium, Low]
 
    Provide a short recommended action.

    Provide a confidence score from 0 to 100.

    """

    try:

        response = client.models.generate_content(

            model="gemini-3-flash-preview",

            contents=prompt,

            config=types.GenerateContentConfig(

                response_mime_type="application/json",

                safety_settings=[

                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},

                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}

                ],

                response_schema={

                    "type": "OBJECT",

                    "properties": {

                        "incident_type": {"type": "STRING"},

                        "urgency_level": {"type": "STRING"},

                        "recommended_action": {"type": "STRING"},

                        "confidence": {"type": "INTEGER"}

                    },

                    "required": ["incident_type", "urgency_level", "recommended_action", "confidence"]

                }

            ),

        )

        if response.text:

            return json.loads(response.text)

        else:

            return {"incident_type": "General Incident", "urgency_level": "Medium", "recommended_action": "Manual review required", "confidence": 0}

    except Exception as e:

        return {"error": str(e)}
 
