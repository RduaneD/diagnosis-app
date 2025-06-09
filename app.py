from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os

app = Flask(__name__)

# Label prediksi
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

# Muat model ML
model = load_model('model.h5')
model.make_predict_function()

# Fungsi prediksi
def predict_label(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds)
    label = label_kelas[predicted_class]
    confidence = float(preds[0][predicted_class]) * 100
    return label, confidence

# Fungsi saran berdasarkan label dan tingkat confidence
def generate_advice(label, confidence):
    if confidence < 60:
        return "Tingkat keyakinan model cukup rendah. Disarankan untuk mengulang diagnosis atau konsultasikan ke ahli."
    elif 60 <= confidence < 85:
        if "sehat" in label:
            return "Tanaman tampak sehat, tetapi tetap perhatikan tanda-tanda penyakit secara berkala."
        elif "sakit" in label:
            return "Tanaman kemungkinan sakit, namun periksa ulang untuk memastikan sebelum melakukan tindakan."
    else:  # confidence >= 85
        if "sehat" in label:
            return "Tanaman Anda terlihat sehat, lanjutkan perawatan seperti biasa."
        elif "sakit" in label:
            return "Tanaman Anda kemungkinan sakit. Periksa gejala lebih lanjut dan lakukan perawatan segera."
    return "Tidak dapat memberikan saran."

# Endpoint diagnosis
@app.route("/api/diagnosis", methods=["POST"])
def api_diagnosis():
    if 'my_image' not in request.files:
        return jsonify({"error": "File gambar tidak ditemukan"}), 400

    file = request.files['my_image']
    if file.filename == '':
        return jsonify({"error": "Nama file kosong"}), 400

    # Pastikan folder uploads/ ada
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    try:
        label, confidence = predict_label(file_path)
        response = {
            "label": label,
            "confidence": round(confidence, 2),
            "advice": generate_advice(label, confidence)
        }
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": f"Terjadi kesalahan saat memproses gambar: {str(e)}"}), 500

    os.remove(file_path)  # Hapus setelah diprediksi
    return jsonify(response)

# Jalankan server
if __name__ == '__main__':
    app.run(debug=True)
