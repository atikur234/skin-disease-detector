import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

# Ensure scripts folder is discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from scripts.inference import SkinClassifier
from llm_advisor import get_llm_advice

# Initialize FastAPI
app = FastAPI(
    title="Skin Disease AI Advisor",
    description="Real-time CNN classification with LLM reasoning."
)

# Configuration for the 10-class Ismail Promus model
MODEL_PATH = os.path.join(current_dir, "weights", "skin_model.pth")
NUM_CLASSES = 10 
CLASSES = [
    "Eczema", "Warts", "Melanoma", "Atopic Dermatitis", "Basal Cell Carcinoma",
    "Melanocytic Nevi", "Benign Keratosis", "Psoriasis", "Seborrheic Keratosis", "Tinea"
]

# Global classifier instance for "Real-time" speed
try:
    classifier = SkinClassifier(MODEL_PATH, num_classes=NUM_CLASSES)
    print("✅ Inference Engine: Ready (10-class MobileNetV3)")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": classifier is not None}

@app.post("/analyze_skin")
async def analyze_skin(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image.")

    try:
        # 1. Read Image
        image_bytes = await file.read()
        
        # 2. CNN Prediction (96.7% Accuracy Model)
        class_idx, confidence = classifier.predict(image_bytes)
        disease_name = CLASSES[class_idx] if class_idx < len(CLASSES) else "UnknownCondition"

        # 3. LLM Reasoning (Gemini 2.5 Flash)
        # This module uses the REST bypass to avoid gRPC DLL errors
        ai_analysis = get_llm_advice(disease_name, confidence)

        return {
            "prediction": disease_name,
            "confidence": round(float(confidence), 4),
            "ai_advisor": ai_analysis,
            "provider": "Gemini 2.5 Flash"
        }
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal analysis failure.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)