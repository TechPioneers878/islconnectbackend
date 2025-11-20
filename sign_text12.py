from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import enchant
from gtts import gTTS
import os
import time
import traceback
import random
import string

from model import load_model_file   # Your model loader

app = Flask(__name__)
CORS(app)

# Load model
model = load_model_file()

# MediaPipe Hands
mp_hands = mp.solutions.hands

# Alphabet mapping
alphabet = ['1','2','3','4','5','6','7','8','9'] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# English dictionary
d = enchant.Dict("en_US")

# Audio folder
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

def get_suggestions(word):
    if len(word) < 2:
        return ["Keep typing..."]
    suggestions = d.suggest(word)
    if not suggestions:
        return ["No suggestions"]
    if word not in suggestions:
        suggestions.insert(0, word)
    return suggestions[:4]


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

        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        # Process with MediaPipe
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

        # Take only first hand
        landmarks = calc_landmark_list(image, results.multi_hand_landmarks[0])
        processed = pre_process_landmark(landmarks)

        # Predict
        preds = model.predict(pd.DataFrame([processed]), verbose=0)
        detected_class = alphabet[int(np.argmax(preds))]

        return jsonify({
            "detected_class": detected_class,
            "word": detected_class.lower(),
            "suggestions": get_suggestions(detected_class.lower())
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/suggestions", methods=["POST"])
def suggestions():
    data = request.get_json()
    partial = data.get("current_word", "")
    return jsonify({"suggestions": get_suggestions(partial)})


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
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="audio/mpeg")


# -------------------------------------------
# MAIN
# -------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
