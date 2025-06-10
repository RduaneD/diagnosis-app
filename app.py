import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import gdown
import tempfile

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.h5"
MODEL_ID = os.getenv("MODEL_ID")

label_kelas = {
    0: 'bayam_sakit',
    1: 'bayam_sehat',
    2: 'kangkung_sakit',
    3: 'kangkung_sehat',
    4: 'pakcoy_sakit',
    5: 'pakcoy_sehat',
    6: 'sawi_sakit',
    7: 'sawi_sehat',
    8: 'selada_sakit',
    9: 'selada_sehat'
}

def download_model():
    if not os.path.exists(MODEL_PATH):
        if not MODEL_ID:
            raise ValueError("MODEL_ID tidak ditemukan.")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

try:
    download_model()
    model = load_model(MODEL_PATH)
    model.make_predict_function()
except Exception as e:
    model = None
    print(f"Gagal load model: {e}")

def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds)
    confidence = float(preds[0][predicted_class]) * 100
    return label_kelas[predicted_class], confidence

def generate_advice(label, confidence):
    if confidence < 60:
        return "Tingkat keyakinan rendah. Ulangi diagnosis atau konsultasi ke ahli."
    elif 60 <= confidence < 85:
        return "Tanaman tampak sehat." if "sehat" in label else "Tanaman kemungkinan sakit, cek ulang."
    else:
        return "Tanaman sehat." if "sehat" in label else "Tanaman sakit. Lakukan tindakan segera."

@app.route('/')
def index():
    return jsonify({"message": "✅ API aktif. Gunakan POST /api/diagnosis."})

@app.route('/api/diagnosis', methods=["POST"])
def diagnose():
    if model is None:
        return jsonify({"error": "Model belum siap."}), 503
    if 'my_image' not in request.files:
        return jsonify({"error": "File gambar tidak ditemukan"}), 400

    file = request.files['my_image']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400

    try:
        tmp = tempfile.gettempdir()
        path = os.path.join(tmp, secure_filename(file.filename))
        file.save(path)

        label, confidence = predict_label(path)
        os.remove(path)

        return jsonify({
            "label": label,
            "confidence": round(confidence, 2),
            "advice": generate_advice(label, confidence)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
