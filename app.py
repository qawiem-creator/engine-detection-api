from flask import Flask, request, jsonify
import numpy as np
import librosa
import joblib
import os
import tempfile

# ─────────────────────────────────────────
# 1. Create Flask app
# ─────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────
# 2. Load your trained model files
# ─────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")

svm     = joblib.load(os.path.join(MODEL_DIR, "svm_engine_model.pkl"))
scaler  = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le      = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("✅ All models loaded successfully!")

# ─────────────────────────────────────────
# 3. Feature extraction function
#    (same as your training code)
# ─────────────────────────────────────────
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

# ─────────────────────────────────────────
# 4. Routes
# ─────────────────────────────────────────

# Route 1: Health check — test if API is running
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "running",
        "message": "Car Engine Fault Detection API is online!"
    })

# Route 2: Main prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Step A: Check if audio file was sent
    if 'file' not in request.files:
        return jsonify({
            "error": "No file uploaded. Send audio file with key 'file'"
        }), 400

    file = request.files['file']

    # Step B: Check file is not empty
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Step C: Check file format
    allowed = {'wav', 'mp3', 'mp4', 'ogg', 'm4a'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({
            "error": f"File type '.{ext}' not supported. Use: {allowed}"
        }), 400

    try:
        # Step D: Save uploaded file temporarily
        temp_dir  = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"audio.{ext}")
        file.save(temp_path)

        # Step E: Convert to wav if needed
        wav_path = os.path.join(temp_dir, "audio.wav")
        if ext != 'wav':
            os.system(f'ffmpeg -i "{temp_path}" -ac 1 -ar 22050 "{wav_path}" -loglevel quiet')
        else:
            wav_path = temp_path

        # Step F: Extract features
        features = extract_features(wav_path)
        features = features.reshape(1, -1)           # shape: (1, 33)
        features_scaled = scaler.transform(features) # normalize

        # Step G: Predict
        prediction_encoded = svm.predict(features_scaled)
        prediction_label   = le.inverse_transform(prediction_encoded)[0]
        confidence         = svm.predict_proba(features_scaled).max() * 100

        # Step H: Build response
        result = {
            "prediction" : prediction_label,           # "normal" or "faulty"
            "confidence" : round(float(confidence), 2),# e.g. 94.21
            "status"     : "🟢 Normal" if prediction_label == "normal" else "🔴 Faulty",
            "message"    : "Engine sounds healthy." if prediction_label == "normal"
                           else "Engine fault detected! Please inspect your vehicle."
        }

        # Step I: Cleanup temp files
        os.remove(temp_path)
        if temp_path != wav_path:
            os.remove(wav_path)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────
# 5. Run the server
# ─────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)