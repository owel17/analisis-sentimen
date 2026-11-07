"""
Lokasi file dan pengaturan yang dipakai bersama seluruh tahap penelitian.

File ini tidak mengolah data. Fungsinya supaya pelabelan, preprocessing,
pelatihan, evaluasi, dan dashboard membaca serta menulis file yang sama.
"""

from pathlib import Path

AKAR = Path(__file__).resolve().parents[1]

NAMA_FILE_MENTAH = "dataset_Google-Maps-Reviews-Scraper_2026-11-04_16-44-52-438.csv"
PATH_MENTAH = AKAR / "dataset" / NAMA_FILE_MENTAH
PATH_BERLABEL = AKAR / "dataset" / "data_berlabel.csv"
PATH_PREPROCESSING = AKAR / "dataset" / "data_preprocessing.csv"
PATH_LATIH = AKAR / "dataset" / "data_latih.csv"
PATH_UJI = AKAR / "dataset" / "data_uji.csv"
PATH_UJI_PREDIKSI = AKAR / "dataset" / "data_uji_prediksi.csv"
# Ulasan yang diketik pengunjung. Skrip pelatihan tidak membaca berkas ini.
PATH_ULASAN_PENGGUNA = AKAR / "dataset" / "ulasan_pengguna.csv"

PATH_MODEL = AKAR / "models" / "svm_sentimen.joblib"
PATH_TFIDF = AKAR / "models" / "tfidf_vectorizer.joblib"

PATH_RINGKASAN_DATA = AKAR / "hasil" / "ringkasan_data.json"
PATH_RINGKASAN_LATIH = AKAR / "hasil" / "ringkasan_pelatihan.json"
PATH_METRIK = AKAR / "hasil" / "metrik_evaluasi.json"
PATH_LAPORAN_KELAS = AKAR / "hasil" / "classification_report.txt"
PATH_KATA = AKAR / "hasil" / "kata_penting_tfidf.csv"
PATH_CONTOH_PREPROCESSING = AKAR / "hasil" / "contoh_preprocessing.txt"
PATH_GAMBAR_SENTIMEN = AKAR / "hasil" / "distribusi_sentimen.png"
PATH_GAMBAR_RATING = AKAR / "hasil" / "distribusi_rating.png"
PATH_GAMBAR_CM = AKAR / "hasil" / "confusion_matrix.png"

KOLOM_TEKS = "text"
KOLOM_BERSIH = "teks_preprocessing"
KOLOM_LABEL = "label_sentimen"
KOLOM_STARS = "stars"

RANDOM_STATE = 42
TEST_SIZE = 0.2
# Unigram dan bigram. Bigram membantu frasa negasi seperti "tidak bagus"
# tetap terlihat oleh TF-IDF.
NGRAM_RANGE = (1, 2)
KERNEL_SVM = "linear"

# Kata ini tidak boleh dihapus saat stopword removal maupun diubah saat stemming.
KATA_NEGASI = frozenset({"tidak", "bukan", "belum", "kurang"})
LABEL_URUTAN = ["negatif", "positif"]
