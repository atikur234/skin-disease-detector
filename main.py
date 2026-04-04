import sys
import os
import time  # New import for timing
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
import uvicorn

# Ensure scripts folder is discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from scripts.inference import SkinClassifier
from llm_advisor import get_llm_advice

app = FastAPI(title="Skin Disease AI Advisor")

# --- ⏱️ REAL-TIME MONITORING MIDDLEWARE ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    # This prints to your VS Code terminal in real-time
    print(f"🚀 [API Perf] {request.method} {request.url.path} | Duration: {process_time:.4f}s")
    # Also adds it to the browser headers for professional debugging
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Configuration
MODEL_PATH = os.path.join(current_dir, "weights", "skin_model.pth")
NUM_CLASSES = 10 
CLASSES = ["Eczema", "Warts", "Melanoma", "Atopic Dermatitis", "Basal Cell Carcinoma", 
           "Melanocytic Nevi", "Benign Keratosis", "Psoriasis", "Seborrheic Keratosis", "Tinea"]

classifier = SkinClassifier(MODEL_PATH, num_classes=NUM_CLASSES)

@app.post("/analyze_skin")
async def analyze_skin(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        class_idx, confidence = classifier.predict(image_bytes)
        disease_name = CLASSES[class_idx]
        
        # This is the "Reasoning" phase that usually takes the most time
        ai_analysis = get_llm_advice(disease_name, confidence)

        return {
            "prediction": disease_name,
            "confidence": round(float(confidence), 4),
            "ai_advisor": ai_analysis
        }
    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal analysis failure.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)