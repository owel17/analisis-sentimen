"""
Tahap 7 dan 8 — Evaluasi model dan visualisasi.

Angka dihitung dari prediksi model yang sudah disimpan
terhadap data uji yang sudah disimpan. Tidak ada angka yang diisi manual.
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from konfigurasi import (
    KOLOM_BERSIH,
    KOLOM_LABEL,
    KOLOM_STARS,
    LABEL_URUTAN,
    PATH_BERLABEL,
    PATH_GAMBAR_CM,
    PATH_GAMBAR_RATING,
    PATH_GAMBAR_SENTIMEN,
    PATH_LAPORAN_KELAS,
    PATH_METRIK,
    PATH_MODEL,
    PATH_TFIDF,
    PATH_UJI,
    PATH_UJI_PREDIKSI,
)


def _ke_python(objek):
    """Ubah angka numpy menjadi angka Python biasa supaya bisa disimpan sebagai JSON."""
    if isinstance(objek, dict):
        return {str(kunci): _ke_python(nilai) for kunci, nilai in objek.items()}
    if isinstance(objek, (list, tuple)):
        return [_ke_python(nilai) for nilai in objek]
    if hasattr(objek, "item"):
        return objek.item()
    return objek


def _simpan_diagram_batang(nilai: pd.Series, judul: str, sumbu_x: str, tujuan) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    warna = ["#9f1239", "#c2410c", "#0f766e", "#115e59", "#134e4a"]
    batang = ax.bar(
        [str(nama) for nama in nilai.index],
        nilai.to_numpy(),
        color=warna[: len(nilai)],
    )
    ax.set_title(judul)
    ax.set_xlabel(sumbu_x)
    ax.set_ylabel("Jumlah ulasan")
    ax.bar_label(batang, padding=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(tujuan, dpi=150)
    plt.close(fig)


def main() -> None:
    for jalur in (PATH_UJI, PATH_MODEL, PATH_TFIDF, PATH_BERLABEL):
        if not jalur.exists():
            raise FileNotFoundError(f"Berkas yang dibutuhkan belum ada: {jalur}")

    uji = pd.read_csv(PATH_UJI, encoding="utf-8-sig")
    model = joblib.load(PATH_MODEL)
    vectorizer = joblib.load(PATH_TFIDF)
    x_uji = vectorizer.transform(uji[KOLOM_BERSIH].fillna("").astype(str))
    prediksi = model.predict(x_uji)
    y_benar = uji[KOLOM_LABEL].astype(str)

    laporan_dict = classification_report(
        y_benar,
        prediksi,
        labels=LABEL_URUTAN,
        output_dict=True,
        zero_division=0,
        digits=4,
    )
    laporan_teks = classification_report(
        y_benar,
        prediksi,
        labels=LABEL_URUTAN,
        zero_division=0,
        digits=4,
    )
    matriks = confusion_matrix(y_benar, prediksi, labels=LABEL_URUTAN)

    metrik = {
        "accuracy": float(accuracy_score(y_benar, prediksi)),
        "precision_macro": float(
            precision_score(y_benar, prediksi, average="macro", zero_division=0, labels=LABEL_URUTAN)
        ),
        "recall_macro": float(
            recall_score(y_benar, prediksi, average="macro", zero_division=0, labels=LABEL_URUTAN)
        ),
        "f1_macro": float(
            f1_score(y_benar, prediksi, average="macro", zero_division=0, labels=LABEL_URUTAN)
        ),
        "precision_weighted": float(
            precision_score(
                y_benar, prediksi, average="weighted", zero_division=0, labels=LABEL_URUTAN
            )
        ),
        "recall_weighted": float(
            recall_score(y_benar, prediksi, average="weighted", zero_division=0, labels=LABEL_URUTAN)
        ),
        "f1_weighted": float(
            f1_score(y_benar, prediksi, average="weighted", zero_division=0, labels=LABEL_URUTAN)
        ),
        "label_urutan": LABEL_URUTAN,
        "confusion_matrix": matriks.tolist(),
        "classification_report": _ke_python(laporan_dict),
        "jumlah_data_uji": int(len(uji)),
    }

    PATH_METRIK.parent.mkdir(parents=True, exist_ok=True)
    PATH_METRIK.write_text(json.dumps(metrik, indent=2, ensure_ascii=False), encoding="utf-8")
    PATH_LAPORAN_KELAS.write_text(laporan_teks, encoding="utf-8")

    hasil_uji = uji.copy()
    hasil_uji["prediksi"] = prediksi
    hasil_uji.to_csv(PATH_UJI_PREDIKSI, index=False, encoding="utf-8-sig")

    berlabel = pd.read_csv(PATH_BERLABEL, encoding="utf-8-sig")
    distribusi_sentimen = (
        berlabel[KOLOM_LABEL].value_counts().reindex(LABEL_URUTAN).fillna(0).astype(int)
    )
    distribusi_rating = berlabel[KOLOM_STARS].value_counts().sort_index()
    _simpan_diagram_batang(
        distribusi_sentimen,
        "Distribusi sentimen data klasifikasi",
        "Sentimen",
        PATH_GAMBAR_SENTIMEN,
    )
    _simpan_diagram_batang(
        distribusi_rating,
        "Distribusi rating data klasifikasi",
        "Bintang",
        PATH_GAMBAR_RATING,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matriks,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=LABEL_URUTAN,
        yticklabels=LABEL_URUTAN,
        ax=ax,
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title("Confusion matrix data uji")
    fig.tight_layout()
    fig.savefig(PATH_GAMBAR_CM, dpi=150)
    plt.close(fig)

    print("Tahap 7 dan 8 selesai.")
    print(f"Accuracy: {metrik['accuracy']:.4f}")
    print(f"Precision macro: {metrik['precision_macro']:.4f}")
    print(f"Recall macro: {metrik['recall_macro']:.4f}")
    print(f"F1-score macro: {metrik['f1_macro']:.4f}")
    print("")
    print(laporan_teks)
    print("Confusion matrix (baris = label sebenarnya, kolom = prediksi):")
    print(f"Urutan label: {LABEL_URUTAN}")
    print(matriks)
    print(f"Metrik disimpan di: {PATH_METRIK}")


if __name__ == "__main__":
    main()
