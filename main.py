import sys
import os
import time
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import uvicorn

# Ensure scripts folder is discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from scripts.inference import SkinClassifier
from llm_advisor import get_llm_advice

app = FastAPI(title="Skin Disease AI Advisor - High Performance")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"⏱️ [API Perf] {request.url.path} | {process_time:.4f}s")
    return response

# Load 10-class model
MODEL_PATH = os.path.join(current_dir, "weights", "skin_model.pth")
classifier = SkinClassifier(MODEL_PATH, num_classes=10)
CLASSES = ["Eczema", "Warts", "Melanoma", "Atopic Dermatitis", "Basal Cell Carcinoma", 
           "Melanocytic Nevi", "Benign Keratosis", "Psoriasis", "Seborrheic Keratosis", "Tinea"]

@app.post("/analyze_skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        # 🚀 LATENCY FIX: Resize image immediately to reduce processing load
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB and resize to what the model actually expects (224x224)
        img = img.convert("RGB").resize((224, 224))
        
        # Convert back to bytes for the classifier
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        optimized_image = img_byte_arr.getvalue()

        # 1. Prediction (Local - Very Fast)
        class_idx, confidence = classifier.predict(optimized_image)
        disease_name = CLASSES[class_idx]

        # 2. Advice (Remote - The Optimized Part)
        ai_analysis = get_llm_advice(disease_name, confidence)

        return {
            "prediction": disease_name,
            "confidence": round(float(confidence), 4),
            "ai_advisor": ai_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Inference failure.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)