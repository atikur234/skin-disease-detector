import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 🚀 PERFORMANCE FIX: Use a persistent Session to keep the connection to Google open
# This saves ~500ms to 1s on every request by avoiding a new SSL handshake.
session = requests.Session()

def get_llm_advice(disease_name, confidence):
    if not GEMINI_API_KEY:
        return {"error": "API Key missing."}

    # Standardized 2026 Endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY.strip()}"
    
    # 🚀 LATENCY FIX: Shorter prompt + JSON constraint = Faster Generation
    prompt = (
        f"Dermatology AI Analysis: {disease_name} ({confidence*100:.1f}% confidence). "
        f"Return ONLY a JSON object with: 'recommendations', 'next_steps', 'tips'. "
        f"Keep advice concise and include a short medical disclaimer."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "max_output_tokens": 300, # 🚀 Limit tokens for speed
            "temperature": 0.1         # 🚀 Low temperature = faster, more direct
        }
    }
    
    headers = {'Content-Type': 'application/json'}

    try:
        # Use session.post instead of requests.post
        response = session.post(url, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        raw_output = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(raw_output)

    except Exception as e:
        print(f"⚠️ LLM Latency/Error: {str(e)[:50]}")
        return {
            "recommendations": ["Apply moisturizer.", "Avoid triggers."],
            "next_steps": ["Consult a dermatologist."],
            "tips": ["Do not scratch."],
            "disclaimer": "Advisor in fallback mode due to latency."
        }