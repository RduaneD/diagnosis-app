
# 🌱 HydroSmart ML Web App

Aplikasi web berbasis Flask untuk mendeteksi kondisi kesehatan tanaman hidroponik dari gambar menggunakan model deep learning.

---

## 🧠 Tujuan Proyek

Membantu petani hidroponik mengidentifikasi apakah tanaman seperti bayam, kangkung, pakcoy, sawi, dan selada dalam kondisi **sehat atau sakit** hanya dengan mengunggah gambar.

---

## 🛠️ Teknologi yang Digunakan

| Komponen       | Teknologi / Library         |
|----------------|------------------------------|
| Backend        | Flask                        |
| Machine Learning | TensorFlow & Keras          |
| Image Processing | Pillow (`PIL`)             |
| Bahasa         | Python                       |
| Deployment     | GitHub (public repo)         |

---

## 🗂️ Struktur File

```
Hydro-Smart-ML-Team/
│
├── app.py                # File utama aplikasi Flask
├── model.h5              # Model Keras terlatih
├── requirements.txt      # Daftar dependensi Python
├── templates/
│   └── index.html        # Antarmuka pengguna (UI)
├── static/
│   └── [uploaded images] # Menyimpan gambar hasil upload
```

---

## 🔍 Cara Kerja Aplikasi

1. Pengguna membuka halaman utama (`/`)
2. Mengunggah gambar tanaman
3. Aplikasi akan:
   - Menyimpan gambar
   - Memproses dan normalisasi gambar
   - Menggunakan model deep learning untuk memprediksi kelas
   - Mengembalikan label dan confidence
4. Hasil ditampilkan ke pengguna:
   - Nama tanaman & status (sehat/sakit)
   - Confidence (keyakinan model)
   - Gambar hasil upload

---

## 🧪 Label Kelas

Model dilatih untuk mengenali 10 label berikut:

| Kode | Label           |
|------|------------------|
| 0    | bayam_sakit      |
| 1    | bayam_sehat      |
| 2    | kangkung_sakit   |
| 3    | kangkung_sehat   |
| 4    | pakcoy_sakit     |
| 5    | pakcoy_sehat     |
| 6    | sawi_sakit       |
| 7    | sawi_sehat       |
| 8    | selada_sakit     |
| 9    | selada_sehat     |

---

## 💡 Contoh Hasil Prediksi

Pengguna akan melihat hasil seperti ini:

- **Label:** `kangkung_sehat`
- **Confidence:** `94.67%`
- **Gambar:** preview gambar yang diunggah

---

## 🚀 Cara Menjalankan (Local)

1. Clone repo:
   ```bash
   git clone https://github.com/HydroSmart-144/Hydro-Smart-ML-Team.git
   cd Hydro-Smart-ML-Team
   ```

2. (Opsional) Buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

4. Jalankan aplikasi:
   ```bash
   python app.py
   ```

5. Buka browser dan kunjungi:
   ```
   http://127.0.0.1:5000/
   ```

---

## 📝 Catatan Tambahan

- Pastikan file `model.h5` ada di direktori utama (`root directory`)
- Gambar hasil upload disimpan di folder `static/`
- Untuk deployment (contoh: Heroku/Render), pastikan tambahkan:
  - `Procfile`
  - `runtime.txt`
  - `requirements.txt` sudah lengkap

---
