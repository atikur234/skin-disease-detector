import os
import json
import requests
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_llm_advice(disease_name, confidence):
    """
    Upgraded for 2026: Uses Gemini 2.5 Flash for high-speed dermatological reasoning.
    """
    if not GEMINI_API_KEY:
        return {"error": "API Key missing in .env"}

    # Use Gemini 2.5 Flash - the standard low-latency model for 2026
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY.strip()}"
    
    prompt_text = (
        f"You are a professional dermatological AI assistant. "
        f"An image analysis model detected '{disease_name}' with {confidence*100:.1f}% confidence. "
        f"Provide a structured response in VALID JSON format with keys: "
        f"'recommendations', 'next_steps', and 'tips'. Include a medical disclaimer."
    )

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, json=payload)
        
        # Diagnostic print if it still fails
        if response.status_code != 200:
            print(f"❌ API Error {response.status_code}: {response.text}")
            # Fallback to older stable model if 2.5 isn't enabled yet
            if "not found" in response.text.lower():
                print("🔄 Attempting fallback to gemini-1.5-flash-latest...")
                url = url.replace("gemini-2.5-flash", "gemini-1.5-flash-latest")
                response = requests.post(url, headers=headers, json=payload)

        response.raise_for_status()
        
        result = response.json()
        raw_output = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(raw_output)

    except Exception as e:
        # Secure masking
        print(f"LLM Advisor Error: Model synchronization issue. Check Google Cloud Console.")
        return {
            "recommendations": "Consult a dermatologist for a professional exam.",
            "next_steps": "Schedule a physical checkup.",
            "tips": "Do not apply unprescribed ointments.",
            "disclaimer": "AI Advisor Offline: Manual consultation required."
        }

if __name__ == "__main__":
    print("🚀 Running 2026 Model Test...")
    advice = get_llm_advice("Eczema", 0.95)
    print(json.dumps(advice, indent=4))