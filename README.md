# Analisis Sentimen Ulasan Sari Ater

Penelitian analisis sentimen ulasan Google Maps Pemandian Air Panas Sari Ater memakai text preprocessing, TF-IDF, dan SVM.

Dataset CSV di `dataset/` telah disanitasi: nama, ID, URL profil/foto, lokasi rinci,
dan metadata reviewer dihapus. Data ulasan turunan tidak menyimpan ID reviewer.
Teks ulasan tetap dipakai untuk analisis; alamat email, nomor telepon, dan URL pada
teks disamarkan. Jangan mengganti dataset ini dengan data yang belum disanitasi
sebelum dipublikasikan.

## Instalasi

Buka terminal di folder proyek ini, lalu jalankan:

```powershell
python -m pip install -r requirements.txt
```

## Menjalankan tahapan penelitian

Jalankan berurutan. Setiap tahap memakai hasil tahap sebelumnya.

```powershell
python src/audit_dataset.py
python src/labeling.py
python src/preprocessing.py
python src/modeling.py
python src/evaluation.py
```

Fungsi tiap script:

- `src/konfigurasi.py` menyimpan lokasi file yang dipakai bersama, supaya setiap tahap membaca file yang sama.
- `src/audit_dataset.py` memeriksa CSV mentah dan menulis `hasil/laporan_audit_dataset.txt`.
- `src/labeling.py` memilih ulasan bahasa Indonesia, menangani duplikat, memberi label, dan menulis `dataset/data_berlabel.csv`.
- `src/preprocessing.py` membersihkan teks dan menulis `dataset/data_preprocessing.csv`.
- `src/modeling.py` membagi data 80:20, membuat TF-IDF dari data latih, melatih SVM, dan menyimpan model di `models/`.
- `src/evaluation.py` menghitung metrik pada data uji dan menyimpan gambar di `hasil/`.

## Menjalankan dashboard

Setelah kelima perintah di atas berhasil:

```powershell
python -m streamlit run app.py
```

Browser akan membuka aplikasi. Jika tidak, buka alamat yang muncul di terminal, biasanya `http://localhost:8501`.

`app.py` hanya membaca dataset dan model yang sudah jadi. Dashboard tidak melatih ulang model.
