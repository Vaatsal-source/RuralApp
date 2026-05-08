CareSathi: AI-Powered Rural Healthcare Suite
CareSathi is an integrated health-tech solution designed to assist healthcare workers in rural or low-resource settings. It leverages local AI models to provide triage, visual diagnostics, and inventory management without requiring a constant high-speed internet connection.

🚀 Key Features
Smart Triage Assistant: Analyzes patient symptoms and vitals (Temp, SpO2, BP) using a local LLM to provide instant diagnostic suggestions and risk levels.

Voice-to-Text Integration: Supports hands-free data entry through multilingual voice transcription (powered by Vosk).

AI Visual Diagnostics: Automated pattern recognition for Chest X-Rays (grayscale) and Skin Lesions (color) using computer vision.

Live Inventory Manager: Real-time tracking and updating of essential medical stock like Paracetamol, ORS, and antibiotics.

Multilingual Support: Fully localized interface available in English, Hindi (हिंदी), Odia (ଓଡ଼ିଆ), and Urdu (اردو).

Referral Generation: Instant PDF generation for patient referrals to higher medical facilities.

🛠️ Technical Architecture
Frontend
Languages: HTML5, CSS3, JavaScript (Vanilla).

Styling: Modern, responsive UI with a floating navigation system and localized text rendering.

Libraries: jsPDF for generating medical reports.

Backend
Framework: Python Flask with Flask-CORS.

AI Engines:

LLM: TinyLlama-1.1B (GGUF format) via llama-cpp-python.

Speech Recognition: Vosk (Small Hindi model) for offline transcription.

Computer Vision: OpenCV and NumPy for image analysis.

Storage: JSON-based local file storage for inventory and browser LocalStorage for the patient queue.

📦 Installation & Setup
1. Prerequisites
Python 3.8+

Web Browser (Chrome/Edge recommended for Voice API support)

2. Backend Setup
Install dependencies:

Bash
pip install flask flask-cors llama-cpp-python vosk numpy opencv-python
Model Placement: Create a models/ folder and place the following:

tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf.

Unzipped Vosk model folder named vosk-model-small-hi-0.22.

Run the server:

Bash
python server.py
3. Frontend Setup
Ensure the server.py is running on http://127.0.0.1:5005.

Open index.html in your browser to access the CareSathi Hub.

📂 Project Structure
Plaintext
├── index.html          # Main Dashboard & Patient Queue
├── triage.html         # AI Symptom & Vital Analysis
├── analysis.html       # X-Ray & Skin Scan Processing
├── inventory.html      # Stock Management Interface
├── server.py           # Flask API with AI Model Integration
├── inventory.json      # Local stock database
└── models/             # Local AI models (LLM & Speech)
⚠️ Disclaimer
This project is a diagnostic aid intended for use by trained healthcare personnel. Final medical decisions should always be made by a qualified Medical Officer.
