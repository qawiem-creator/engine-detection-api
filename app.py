from flask import Flask, request, jsonify
import numpy as np
import librosa
import joblib
import os
import tempfile
import subprocess

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

svm    = joblib.load(os.path.join(MODEL_DIR, "svm_engine_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("✅ All models loaded successfully!")

def extract_features(file_path, n_mfcc=11):
    audio, sr = librosa.load(file_path, sr=22050, duration=3.0)
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
            '-ac', '1', '-ar', '22050',
            output_path, '-y', '-loglevel', 'quiet'
        ], check=True, timeout=30)
        return True
    except Exception as e:
        print(f"ffmpeg error: {e}")
        return False

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "message": "Car Engine Fault Detection API is online!"
    })

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

    try:
        temp_dir  = tempfile.mkdtemp()
        # Clean filename - remove spaces
        clean_name = f"audio.{ext}"
        temp_path = os.path.join(temp_dir, clean_name)
        file.save(temp_path)

        print(f"📁 File saved: {temp_path}")
        print(f"📁 File size: {os.path.getsize(temp_path)} bytes")

        # Convert to wav if needed
        wav_path = os.path.join(temp_dir, "audio.wav")
        if ext != 'wav':
            print("🔄 Converting to WAV...")
            success = convert_to_wav(temp_path, wav_path)
            if not success:
                # Try loading directly with librosa
                wav_path = temp_path
                print("⚠️ ffmpeg failed, trying librosa directly...")

        # Extract features
        print("🔍 Extracting features...")
        features = extract_features(wav_path)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Predict
        print("🤖 Predicting...")
        prediction_encoded = svm.predict(features_scaled)
        prediction_label   = le.inverse_transform(prediction_encoded)[0]
        confidence         = float(svm.predict_proba(features_scaled).max() * 100)

        print(f"✅ Result: {prediction_label} ({confidence:.2f}%)")

        # Cleanup
        try:
            os.remove(temp_path)
            if wav_path != temp_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except:
            pass

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)