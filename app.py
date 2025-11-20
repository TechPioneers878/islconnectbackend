from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os
import time
import traceback
import random
import string

from model import load_model_file   # Your model loader

# ---------------------------
# Gemini API (NEW)
# ---------------------------
from google import genai
from google.genai import types
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.0-flash"

# ---------------------------

app = Flask(__name__)
CORS(app)

model = load_model_file()
mp_hands = mp.solutions.hands

alphabet = ['1','2','3','4','5','6','7','8','9'] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

AUDIO_FOLDER = "generated_audio"
os.makedirs(AUDIO_FOLDER, exist_ok=True)
current_audio_file = None


# -------------------------------------------
# Helper functions
# -------------------------------------------

def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[int(lm.x * w), int(lm.y * h)] for lm in landmarks.landmark]


def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    arr = np.array([[x - base_x, y - base_y] for x, y in landmark_list]).flatten()
    max_val = max(abs(arr))
    return arr / max_val if max_val > 0 else arr


# ---------- GEMINI SUGGESTION ENGINE ----------
def gemini_suggest(current_word, sentence):
    """
    Returns both:
    - Word completion (for partial words)
    - Next word prediction (based on sentence)
    """

    prompt = f"""
You are an AI assisting Indian Sign Language recognition.

User has:
Current partial word: "{current_word}"
Current full sentence: "{sentence}"

Task:
1. If current_word is incomplete → suggest best completions.
2. If sentence is long → suggest next-word predictions.
3. Always return 4–6 SHORT suggestions.
4. Only return the words, no explanation.

Return response as:
word1, word2, word3, word4
"""
"""
    try:
        response = gemini_client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt]
        )

        text = ""
        for c in response.candidates:
            for p in c.content.parts:
                if hasattr(p, "text"):
                    text += p.text

        # Convert "word1, word2, word3" into list
        words = [w.strip() for w in text.replace("\n", "").split(",") if w.strip()]

        return words[:6]

    except Exception as e:
        print("Gemini error:", e)
        return ["No suggestions"]


# -------------------------------------------
# API ROUTES
# -------------------------------------------

@app.route("/")
def index():
    return jsonify({"message": "ISL API running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
            results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return jsonify({
                "detected_class": "-",
                "word": "",
                "suggestions": []
            })

        # Only first hand
        landmarks = calc_landmark_list(image, results.multi_hand_landmarks[0])
        processed = pre_process_landmark(landmarks)

        preds = model.predict(pd.DataFrame([processed]), verbose=0)
        detected_class = alphabet[int(np.argmax(preds))]

        return jsonify({
            "detected_class": detected_class,
            "word": detected_class.lower(),
            "suggestions": []   # frontend will call suggestions route
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/suggestions", methods=["POST"])
def suggestions():
    data = request.get_json()
    current_word = data.get("current_word", "")
    sentence = data.get("sentence", "")

    out = gemini_suggest(current_word, sentence)
    return jsonify({"suggestions": out})


@app.route("/speak", methods=["POST"])
def speak():
    global current_audio_file
    try:
        data = request.get_json()
        text = data.get("text", "")
        lang = data.get("language", "en")

        file_id = f"{int(time.time())}_{random.randint(1000,9999)}.mp3"
        file_path = os.path.join(AUDIO_FOLDER, file_id)

        tts = gTTS(text=text, lang=lang)
        tts.save(file_path)

        current_audio_file = file_path
        return jsonify({"audio_url": f"/current_audio?f={file_id}"})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)})


@app.route("/current_audio")
def get_audio():
    f = request.args.get("f")
    if not f:
        return jsonify({"error": "Missing file"}), 404
    path = os.path.join(AUDIO_FOLDER, f)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="audio/mpeg")


# -------------------------------------------
# MAIN SERVER
# -------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
