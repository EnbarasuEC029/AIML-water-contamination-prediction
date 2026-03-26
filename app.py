from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import threading
import time
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import deque

# ── Firebase Import ───────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, db as firebase_db
    FIREBASE_AVAILABLE = True
    print("[Firebase] Package found.")
except ImportError:
    FIREBASE_AVAILABLE = False
    print("[Firebase] NOT installed.")

app = Flask(__name__)

# ───────────────── CONFIG ─────────────────
FIREBASE_DB_URL = "https://aiml-prediction-default-rtdb.firebaseio.com/"
FIREBASE_NODE_PATH = "/waterQuality"

FIREBASE_TDS_KEY = "tds_ppm"
FIREBASE_TURBIDITY_KEY = "turbidity_ntu"

MODEL_PATH = "water_quality_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

POLL_INTERVAL = 5
HISTORY_SIZE = 50

# ───────────────── STATE ─────────────────
history = deque(maxlen=HISTORY_SIZE)
latest = {
    "tds": 0,
    "turbidity": 0,
    "quality": "Unknown",
    "safe": False,
    "confidence": 0,
    "timestamp": "",
    "demo_mode": True
}

# ───────────────── DATASET ─────────────────
EMBEDDED_DATA = """tds_ppm,turbidity_ntu,label
50,0.4,Excellent
100,0.7,Excellent
250,1.5,Good
350,3.0,Good
500,5.0,Fair
700,9.0,Fair
900,15.0,Poor
1100,25.0,Poor
"""

# ───────────────── TRAIN MODEL ─────────────────
def train_model():
    from io import StringIO
    df = pd.read_csv(StringIO(EMBEDDED_DATA))

    X = df[['tds_ppm', 'turbidity_ntu']]
    y = df['label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[ML] Accuracy: {acc*100:.1f}%")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    return model, le

# ───────────────── LOAD MODEL ─────────────────
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
else:
    model, le = train_model()

# ───────────────── PREDICT ─────────────────
SAFE_LABELS = {"Excellent", "Good"}

def predict(tds, turb):
    df = pd.DataFrame([[tds, turb]], columns=['tds_ppm', 'turbidity_ntu'])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]

    label = le.inverse_transform([pred])[0]
    confidence = round(max(prob)*100, 1)

    return label, confidence

# ───────────────── FIREBASE INIT ─────────────────
firebase_connected = False

if FIREBASE_AVAILABLE:
    try:
        key = os.environ.get("FIREBASE_KEY")

        if not key:
            raise ValueError("FIREBASE_KEY missing")

        firebase_json = json.loads(key)
        cred = credentials.Certificate(firebase_json)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': FIREBASE_DB_URL
            })

        firebase_connected = True
        print("[Firebase] Connected")

    except Exception as e:
        print("[Firebase ERROR]", e)

# ───────────────── DEMO ─────────────────
def demo_data():
    return np.random.randint(50,1000), round(np.random.uniform(0.3,15),2)

# ───────────────── POLL LOOP ─────────────────
def poll_loop():
    global latest

    while True:
        try:
            if firebase_connected:
                data = firebase_db.reference(FIREBASE_NODE_PATH).get()

                if not data:
                    raise ValueError("No data")

                tds = float(data.get(FIREBASE_TDS_KEY, 0))
                turb = float(data.get(FIREBASE_TURBIDITY_KEY, 0))
                demo = False
            else:
                tds, turb = demo_data()
                demo = True

            quality, conf = predict(tds, turb)
            safe = quality in SAFE_LABELS

            now = datetime.now().strftime("%H:%M:%S")

            latest = {
                "tds": tds,
                "turbidity": turb,
                "quality": quality,
                "safe": safe,
                "confidence": conf,
                "timestamp": now,
                "demo_mode": demo
            }

            history.append(latest)

            # push back
            if firebase_connected:
                firebase_db.reference(FIREBASE_NODE_PATH).update({
                    "predicted_quality": quality,
                    "is_safe": safe
                })

        except Exception as e:
            print("[Loop Error]", e)

        time.sleep(POLL_INTERVAL)

# Start thread
threading.Thread(target=poll_loop, daemon=True).start()

# ───────────────── ROUTES ─────────────────
@app.route('/')
def home():
    return render_template("dashboard.html")

@app.route('/api/latest')
def api_latest():
    return jsonify(latest)

@app.route('/api/history')
def api_history():
    return jsonify(list(history))

@app.route('/api/status')
def status():
    return jsonify({
        "firebase": firebase_connected
    })

# ───────────────── RUN ─────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)