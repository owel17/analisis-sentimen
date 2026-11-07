"""
Tahap 4, 5, dan 6 — Pembagian data, TF-IDF, dan pelatihan SVM.

TF-IDF di-fit hanya pada data latih.
Data uji hanya di-transform dengan vectorizer yang sama.
Model disimpan di folder models. Evaluasi angka dilakukan oleh evaluation.py.
"""

import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from konfigurasi import (
    KERNEL_SVM,
    KOLOM_BERSIH,
    KOLOM_LABEL,
    NGRAM_RANGE,
    PATH_KATA,
    PATH_LATIH,
    PATH_MODEL,
    PATH_PREPROCESSING,
    PATH_RINGKASAN_LATIH,
    PATH_TFIDF,
    PATH_UJI,
    RANDOM_STATE,
    TEST_SIZE,
)


def _kata_penting(vectorizer: TfidfVectorizer, matriks, label: pd.Series) -> pd.DataFrame:
    """Ambil istilah dengan rata-rata TF-IDF tertinggi pada tiap kelas di data latih."""
    import numpy as np

    nama = vectorizer.get_feature_names_out()
    baris = []
    for nama_kelas in ("negatif", "positif"):
        pilihan = label.to_numpy() == nama_kelas
        if pilihan.sum() == 0:
            continue
        rata = np.asarray(matriks[pilihan].mean(axis=0)).ravel()
        urutan = rata.argsort()[::-1][:15]
        for peringkat, indeks in enumerate(urutan, start=1):
            baris.append(
                {
                    "label": nama_kelas,
                    "peringkat": peringkat,
                    "kata": nama[indeks],
                    "rata_tfidf": float(rata[indeks]),
                }
            )
    return pd.DataFrame(baris)


def main() -> None:
    if not PATH_PREPROCESSING.exists():
        raise FileNotFoundError(
            "Hasil preprocessing belum ada. Jalankan dulu: python src/preprocessing.py"
        )

    data = pd.read_csv(PATH_PREPROCESSING, encoding="utf-8-sig")
    data[KOLOM_BERSIH] = data[KOLOM_BERSIH].fillna("").astype(str)
    kosong = data[KOLOM_BERSIH].str.strip().eq("")
    siap = data.loc[~kosong].copy()
    if siap[KOLOM_LABEL].nunique() < 2:
        raise ValueError("Data tidak memiliki dua kelas. Pelatihan SVM dihentikan.")

    latih, uji = train_test_split(
        siap,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=siap[KOLOM_LABEL],
    )
    latih = latih.reset_index(drop=True)
    uji = uji.reset_index(drop=True)

    vectorizer = TfidfVectorizer(ngram_range=NGRAM_RANGE)
    x_latih = vectorizer.fit_transform(latih[KOLOM_BERSIH])
    # Data uji tidak ikut fit. Baris ini hanya memastikan transform bisa dipakai.
    vectorizer.transform(uji[KOLOM_BERSIH])

    model = SVC(kernel=KERNEL_SVM, random_state=RANDOM_STATE)
    model.fit(x_latih, latih[KOLOM_LABEL])

    PATH_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATH_MODEL)
    joblib.dump(vectorizer, PATH_TFIDF)
    latih.to_csv(PATH_LATIH, index=False, encoding="utf-8-sig")
    uji.to_csv(PATH_UJI, index=False, encoding="utf-8-sig")

    kata = _kata_penting(vectorizer, x_latih, latih[KOLOM_LABEL])
    PATH_KATA.parent.mkdir(parents=True, exist_ok=True)
    kata.to_csv(PATH_KATA, index=False, encoding="utf-8-sig")

    def hitung(bagian: pd.DataFrame, label: str) -> int:
        return int(bagian[KOLOM_LABEL].eq(label).sum())

    ringkasan = {
        "baris_masuk": int(len(data)),
        "baris_kosong_setelah_preprocessing": int(kosong.sum()),
        "baris_dipakai": int(len(siap)),
        "jumlah_latih": int(len(latih)),
        "jumlah_uji": int(len(uji)),
        "positif_latih": hitung(latih, "positif"),
        "negatif_latih": hitung(latih, "negatif"),
        "positif_uji": hitung(uji, "positif"),
        "negatif_uji": hitung(uji, "negatif"),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": True,
        "ngram_range": list(NGRAM_RANGE),
        "jumlah_fitur_tfidf": int(len(vectorizer.get_feature_names_out())),
        "kernel_svm": KERNEL_SVM,
        "class_weight": None,
        "file_model": PATH_MODEL.name,
        "file_tfidf": PATH_TFIDF.name,
    }
    PATH_RINGKASAN_LATIH.write_text(
        json.dumps(ringkasan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Tahap 4–6 selesai. Model dilatih hanya pada data latih.")
    print(json.dumps(ringkasan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
