from flask import Flask, request, jsonify
import numpy as np
import librosa
import joblib
import os
import tempfile
import subprocess
import time                                          # ← add this

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

SERVER_START_TIME = time.time()                      # ← add this

svm    = joblib.load(os.path.join(MODEL_DIR, "svm_engine_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("✅ All models loaded successfully!")

def extract_features(file_path, n_mfcc=11):
    audio, sr = librosa.load(file_path, sr=16000, duration=5.0, mono=True)
    mfcc   = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        np.mean(mfcc,   axis=1),
        np.mean(delta,  axis=1),
        np.mean(delta2, axis=1)
    ])

def convert_to_wav(input_path, output_path):
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-ac', '1', '-ar', '16000',
            '-t', '5',
            output_path, '-y', '-loglevel', 'quiet'
        ], check=True, timeout=30)
        return True
    except Exception as e:
        print(f"ffmpeg error: {e}")
        return False

def cleanup_files(*paths):
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "message": "Car Engine Fault Detection API is online!"
    })

@app.route('/ping', methods=['GET'])                 # ← moved here, before main
def ping():
    uptime = round(time.time() - SERVER_START_TIME, 1)
    return jsonify({"status": "awake", "uptime_seconds": uptime}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit('.', 1)[-1].lower()
    allowed = {'wav', 'mp3', 'mp4', 'ogg', 'm4a'}
    if ext not in allowed:
        return jsonify({"error": f"File type .{ext} not supported"}), 400

    temp_dir  = None
    temp_path = None
    wav_path  = None

    try:
        temp_dir   = tempfile.mkdtemp()
        clean_name = f"audio.{ext}"
        temp_path  = os.path.join(temp_dir, clean_name)
        file.save(temp_path)

        print(f"📁 File saved: {temp_path}")
        print(f"📁 File size: {os.path.getsize(temp_path)} bytes")

        wav_path = os.path.join(temp_dir, "audio.wav")
        if ext != 'wav':
            print("🔄 Converting to WAV...")
            success = convert_to_wav(temp_path, wav_path)
            if not success:
                print("⚠️ ffmpeg failed, trying librosa directly...")
                wav_path = temp_path
        else:
            print("🔄 Normalising WAV...")
            success = convert_to_wav(temp_path, wav_path)
            if not success:
                wav_path = temp_path

        print("🔍 Extracting features...")
        features        = extract_features(wav_path)
        features        = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        print("🤖 Predicting...")
        prediction_encoded = svm.predict(features_scaled)
        prediction_label   = le.inverse_transform(prediction_encoded)[0]
        confidence         = float(svm.predict_proba(features_scaled).max() * 100)

        print(f"✅ Result: {prediction_label} ({confidence:.2f}%)")

        return jsonify({
            "prediction": prediction_label,
            "confidence": round(confidence, 2),
            "status":     "Normal" if prediction_label == "normal" else "Faulty",
            "message":    "Engine sounds healthy." if prediction_label == "normal"
                         else "Engine fault detected!"
        }), 200

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        cleanup_files(temp_path, wav_path)
        if temp_dir and os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except Exception:
                pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)