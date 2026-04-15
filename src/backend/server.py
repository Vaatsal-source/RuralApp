import os
import json
import wave
import sys
import numpy as np
import cv2
from datetime import datetime
from llama_cpp import Llama
from vosk import Model, KaldiRecognizer
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "models")
INVENTORY_FILE = os.path.join(BASE_DIR, "inventory.json")
PATIENTS_FILE = os.path.join(BASE_DIR, "patients.json")

# Model Filenames
LLM_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
VOSK_MODEL_DIR = os.path.join(MODELS_PATH, "vosk-model-small-hi-0.22")

# --- DATA HELPERS ---
def load_json(path, default):
    if not os.path.exists(path):
        with open(path, 'w') as f: json.dump(default, f)
        return default
    with open(path, 'r') as f: 
        try:
            return json.load(f)
        except:
            return default

def save_json(path, data):
    with open(path, 'w') as f: json.dump(data, f, indent=4)

# --- INITIALIZE ENGINES ---
try:
    print("Loading AI Engines...")
    llm = Llama(model_path=os.path.join(MODELS_PATH, LLM_FILE), n_ctx=2048, verbose=False)
    vosk_model = Model(VOSK_MODEL_DIR)
    print("Engines Ready.")
except Exception as e:
    print(f"Engine Load Error: {e}")
    sys.exit(1)

# --- INVENTORY ROUTES ---
@app.route('/get-inventory', methods=['GET'])
def get_inventory():
    return jsonify(load_json(INVENTORY_FILE, {"Paracetamol": 50, "ORS": 100}))

@app.route('/update-inventory', methods=['POST'])
def update_inventory():
    data = request.json
    inventory = load_json(INVENTORY_FILE, {})
    inventory[data.get("item")] = max(0, int(data.get("quantity", 0)))
    save_json(INVENTORY_FILE, inventory)
    return jsonify({"status": "success", "inventory": inventory})

# --- PATIENT MANAGEMENT ROUTES ---
@app.route('/get-patients', methods=['GET'])
def get_patients():
    return jsonify(load_json(PATIENTS_FILE, {}))

@app.route('/delete-patient', methods=['POST'])
def delete_patient():
    data = request.json
    p_id = data.get('id')
    patients = load_json(PATIENTS_FILE, {})
    if p_id in patients:
        del patients[p_id]
        save_json(PATIENTS_FILE, patients)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "ID not found"}), 404

@app.route('/add-patient', methods=['POST'])
def add_patient():
    data = request.json
    patients = load_json(PATIENTS_FILE, {})
    p_id = f"PAT-{np.random.randint(1000, 9999)}"
    patients[p_id] = {
        "name": data.get("name"),
        "age": data.get("age"),
        "village": data.get("village"),
        "history": [],
        "current_vitals": {}
    }
    save_json(PATIENTS_FILE, patients)
    return jsonify({"status": "success", "id": p_id})

@app.route('/update-patient-medical', methods=['POST'])
def update_patient_medical():
    data = request.json
    patients = load_json(PATIENTS_FILE, {})
    p_id = data.get('id')
    if p_id in patients:
        record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "vitals": data.get("vitals"),
            "diagnosis": data.get("diagnosis")
        }
        patients[p_id]['history'].append(record)
        patients[p_id]['current_vitals'] = data.get("vitals")
        save_json(PATIENTS_FILE, patients)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Patient not found"}), 404

# --- AI ANALYSIS ROUTES ---
@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get("text", "")
    vitals = data.get("vitals", {})
    inventory = load_json(INVENTORY_FILE, {})
    available_meds = [name for name, qty in inventory.items() if qty > 0]
    
    prompt = f"""<|system|>
You are a CureBay CareSathi medical assistant. Suggest medicines only from: {', '.join(available_meds)}.
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
    
    # Logic to differentiate scan types
    is_grayscale = np.allclose(img_cv[:,:,0], img_cv[:,:,1], atol=15)
    scan_type = "Chest X-Ray" if is_grayscale else "Skin Lesion/Dermatology"
    
    # NEW: LLM-POWERED DYNAMIC ANALYSIS
    prompt = f"""<|system|>
You are a CureBay CareSathi radiologist and dermatologist AI. 
Generate a professional medical interpretation for a {scan_type} scan.
Mention potential observations (like opacity, congestion, or irregular borders) based on standard clinical knowledge for this scan type.
STRICT RULE: End with 'Please consult a senior doctor for final validation.'</s>
<|user|>
Interpret this {scan_type} scan.</s>
<|assistant|>"""
    
    output = llm(prompt, max_tokens=250, stop=["</s>"], echo=False)
    explanation = output["choices"][0]["text"].strip()
    
    return jsonify({
        "status": "success", 
        "label": f"AI Interpretation: {scan_type}", 
        "explanation": explanation
    })

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