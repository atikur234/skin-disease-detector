# Skin Disease AI Advisor (96.7% Accuracy)

A professional-grade dermatological advisor system that combines a deep learning CNN model with LLM-powered medical reasoning.

## 🚀 Features
* **High-Accuracy Inference:** MobileNetV3 model trained on the Ismail Promus dataset (10-class subset) with 96.7% validation accuracy.
* **AI Reasoning Module:** Integrated with **Gemini 2.5 Flash** to provide real-time recommendations, next steps, and tips.
* **REST API Architecture:** Built with **FastAPI** for high-performance, asynchronous image processing.
* **Production Security:** Implemented environment-based credential management (.env) and log sanitization to protect API keys.

## 🛠️ Tech Stack
* **Backend:** FastAPI, Uvicorn
* **Deep Learning:** PyTorch, Torchvision
* **LLM:** Google Gemini 2.5 Flash (via REST API)
* **Data Handling:** PIL, Requests, Python-Dotenv

## 📦 Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd skin-disease-detector
   python -m venv venv
   .\venv\Scripts\activate
   pip install "numpy<2"
   pip install -r requirements.txt
   python main.py