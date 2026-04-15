import os
import json
import wave
import sys
import numpy as np
import cv2
from llama_cpp import Llama
from vosk import Model, KaldiRecognizer
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "models")
INVENTORY_FILE = os.path.join(BASE_DIR, "inventory.json")

# Model Filenames
LLM_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
VOSK_MODEL_DIR = os.path.join(MODELS_PATH, "vosk-model-small-hi-0.22")

# --- INITIALIZE INVENTORY ---
def load_inventory():
    if not os.path.exists(INVENTORY_FILE):
        default_inv = {"Paracetamol": 50, "ORS": 100, "Cetirizine": 30, "Amoxicillin": 10}
        with open(INVENTORY_FILE, 'w') as f:
            json.dump(default_inv, f)
        return default_inv
    with open(INVENTORY_FILE, 'r') as f:
        return json.load(f)

# --- INITIALIZE ENGINES ---
try:
    print("Loading TinyLlama and Vosk engines...")
    llm = Llama(model_path=os.path.join(MODELS_PATH, LLM_FILE), n_ctx=2048, verbose=False)
    vosk_model = Model(VOSK_MODEL_DIR)
    print("Engines loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: {e}")
    sys.exit(1)

# --- ROUTES ---

@app.route('/get-inventory', methods=['GET'])
def get_inventory():
    return jsonify(load_inventory())

@app.route('/update-inventory', methods=['POST'])
def update_inventory():
    data = request.json
    item = data.get("item")
    quantity = data.get("quantity", 0)
    inventory = load_inventory()
    inventory[item] = max(0, int(quantity))
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f)
    return jsonify({"status": "success", "inventory": inventory})

@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get("text", "")
    vitals = data.get("vitals", {})
    
    inventory = load_inventory()
    # Dynamic constraint: AI only sees what's in stock
    available_meds = [name for name, qty in inventory.items() if qty > 0]
    
    prompt = f"""<|system|>
You are a CureBay CareSathi medical assistant. Diagnose symptoms and provide triage.
STRICT RULE: Only suggest medicines from this available list: {', '.join(available_meds)}.
Format:
CONDITION: [Name]
RISK: [Low/High]
ADVICE: [Next steps]</s>
<|user|>
Vitals: {vitals} | Symptoms: {text}</s>
<|assistant|>"""
    
    output = llm(prompt, max_tokens=300, stop=["</s>"], echo=False)
    return jsonify({"status": "success", "diagnosis": output["choices"][0]["text"].strip()})

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    file = request.files['file']
    img_path = os.path.join(BASE_DIR, "temp_img.jpg")
    file.save(img_path)
    img_cv = cv2.imread(img_path)
    # Detect if X-Ray (Grayscale) or Skin (Color)
    is_grayscale = np.allclose(img_cv[:,:,0], img_cv[:,:,1], atol=15)
    label = "Chest X-Ray Pattern" if is_grayscale else "Skin Lesion Pattern"
    return jsonify({"status": "success", "label": label, "explanation": "AI suggests clinical correlation for localized findings."})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['file']
    audio_path = os.path.join(BASE_DIR, "temp_voice.wav")
    audio_file.save(audio_path)
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    res = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0: break
        if rec.AcceptWaveform(data): res += json.loads(rec.Result())["text"] + " "
    res += json.loads(rec.FinalResult())["text"]
    return jsonify({"status": "success", "transcript": res.strip()})

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5005)