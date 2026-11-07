"""
Dashboard Streamlit untuk hasil penelitian analisis sentimen Sari Ater.

Aplikasi ini hanya menampilkan data, model, dan metrik yang sudah dihasilkan
script di folder src. Tidak ada pelatihan ulang dan tidak ada angka buatan.
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

AKAR = Path(__file__).resolve().parent
sys.path.insert(0, str(AKAR / "src"))

from konfigurasi import (  # noqa: E402
    KOLOM_BERSIH,
    KOLOM_LABEL,
    KOLOM_STARS,
    KOLOM_TEKS,
    LABEL_URUTAN,
    PATH_BERLABEL,
    PATH_KATA,
    PATH_LAPORAN_KELAS,
    PATH_METRIK,
    PATH_MODEL,
    PATH_PREPROCESSING,
    PATH_RINGKASAN_DATA,
    PATH_RINGKASAN_LATIH,
    PATH_TFIDF,
    PATH_UJI_PREDIKSI,
)
from preprocessing import preprocess_teks  # noqa: E402


st.set_page_config(
    page_title="Sentimen Sari Ater",
    page_icon="♨️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _muat_json(jalur: Path) -> dict | None:
    if not jalur.exists():
        return None
    return json.loads(jalur.read_text(encoding="utf-8"))


def _muat_csv(jalur: Path) -> pd.DataFrame | None:
    if not jalur.exists():
        return None
    return pd.read_csv(jalur, encoding="utf-8-sig")


@st.cache_resource
def _muat_model(sidik_model: float, sidik_tfidf: float):
    """Muat model SVM dan TF-IDF yang sudah dilatih. Argumen sidik memakai waktu file."""
    del sidik_model, sidik_tfidf
    return joblib.load(PATH_MODEL), joblib.load(PATH_TFIDF)


def _terapkan_gaya() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f6f4ef; }
        section[data-testid="stSidebar"] { background: #143d3a; }
        section[data-testid="stSidebar"] * { color: #f7f5f0; }
        section[data-testid="stSidebar"] a { color: #f7f5f0; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e6e1d6;
            border-radius: 16px;
            padding: 14px 16px;
        }
        .kotak-hasil {
            border-radius: 18px;
            padding: 22px 24px;
            color: white;
            font-size: 1.6rem;
            font-weight: 650;
        }
        .pengantar {
            color: #3d4f4d;
            font-size: 1.05rem;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _peringatan_berkas(nama: str) -> None:
    st.warning(
        f"Berkas `{nama}` belum ada. Jalankan dulu tahapan penelitian dari terminal, "
        "lalu muat ulang halaman ini."
    )


def halaman_dashboard() -> None:
    st.title("Dashboard")
    st.markdown(
        "<p class='pengantar'>Ringkasan data ulasan Pemandian Air Panas Sari Ater "
        "yang dipakai untuk klasifikasi sentimen.</p>",
        unsafe_allow_html=True,
    )
    ringkasan = _muat_json(PATH_RINGKASAN_DATA)
    data = _muat_csv(PATH_BERLABEL)
    if ringkasan is None or data is None:
        _peringatan_berkas("hasil/ringkasan_data.json dan dataset/data_berlabel.csv")
        return

    atas = st.columns(2)
    bawah = st.columns(2)
    atas[0].metric("Ulasan mentah", f"{ringkasan['jumlah_baris_mentah']:,}".replace(",", "."))
    atas[1].metric("Data klasifikasi", f"{ringkasan['data_klasifikasi']:,}".replace(",", "."))
    bawah[0].metric("Sentimen positif", f"{ringkasan['positif']:,}".replace(",", "."))
    bawah[1].metric("Sentimen negatif", f"{ringkasan['negatif']:,}".replace(",", "."))

    st.caption(
        "Data klasifikasi adalah ulasan bahasa Indonesia yang memiliki teks, "
        "bukan duplikat, dan bintangnya 1, 2, 4, atau 5. "
        f"Bintang 3 yang dikeluarkan: {ringkasan['rating_3_dikeluarkan']}. "
        f"Rentang tanggal: {ringkasan['tanggal_paling_awal']} sampai {ringkasan['tanggal_paling_akhir']}."
    )

    distribusi = data[KOLOM_LABEL].value_counts().reindex(LABEL_URUTAN).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    batang = ax.bar(distribusi.index, distribusi.to_numpy(), color=["#9f1239", "#0f766e"])
    ax.bar_label(batang, padding=3)
    ax.set_title("Distribusi sentimen")
    ax.set_ylabel("Jumlah ulasan")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def halaman_data() -> None:
    st.title("Data ulasan")
    data = _muat_csv(PATH_BERLABEL)
    if data is None:
        _peringatan_berkas("dataset/data_berlabel.csv")
        return

    label = st.multiselect("Label sentimen", LABEL_URUTAN, default=LABEL_URUTAN)
    kata = st.text_input("Cari kata di teks ulasan")
    tampilan = data.loc[data[KOLOM_LABEL].isin(label)].copy()
    if kata.strip():
        tampilan = tampilan.loc[
            tampilan[KOLOM_TEKS].astype(str).str.contains(kata.strip(), case=False, na=False)
        ]

    st.caption(f"Menampilkan {len(tampilan)} dari {len(data)} ulasan berlabel.")
    st.dataframe(
        tampilan[["publishedAtDate", KOLOM_STARS, KOLOM_LABEL, KOLOM_TEKS]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "publishedAtDate": "Tanggal",
            KOLOM_STARS: "Rating",
            KOLOM_LABEL: "Sentimen",
            KOLOM_TEKS: st.column_config.TextColumn("Ulasan", width="large"),
        },
    )


def halaman_preprocessing() -> None:
    st.title("Preprocessing")
    st.markdown(
        "Perbandingan teks asli dan hasil cleansing, case folding, tokenizing, "
        "stopword removal, serta stemming. Kata **tidak**, **bukan**, **belum**, "
        "dan **kurang** tidak dihapus."
    )
    data = _muat_csv(PATH_PREPROCESSING)
    if data is None:
        _peringatan_berkas("dataset/data_preprocessing.csv")
        return

    pilihan_label = st.selectbox("Label", ["semua", *LABEL_URUTAN])
    bagian = data if pilihan_label == "semua" else data.loc[data[KOLOM_LABEL].eq(pilihan_label)]
    if bagian.empty:
        st.info("Tidak ada ulasan pada label ini.")
        return

    indeks = st.number_input(
        "Nomor contoh",
        min_value=0,
        max_value=len(bagian) - 1,
        value=0,
        step=1,
    )
    baris = bagian.iloc[int(indeks)]
    kiri, kanan = st.columns(2)
    kiri.subheader("Teks asli")
    kiri.text_area("asli", baris[KOLOM_TEKS], height=220, label_visibility="collapsed")
    kanan.subheader("Hasil preprocessing")
    kanan.text_area("bersih", baris[KOLOM_BERSIH], height=220, label_visibility="collapsed")
    st.caption(f"Rating {baris[KOLOM_STARS]} · label {baris[KOLOM_LABEL]}")


def halaman_analisis() -> None:
    st.title("Analisis sentimen")
    data = _muat_csv(PATH_BERLABEL)
    kata = _muat_csv(PATH_KATA)
    if data is None:
        _peringatan_berkas("dataset/data_berlabel.csv")
        return

    kiri, kanan = st.columns(2)
    with kiri:
        distribusi = data[KOLOM_LABEL].value_counts().reindex(LABEL_URUTAN).fillna(0).astype(int)
        fig, ax = plt.subplots(figsize=(6, 4))
        batang = ax.bar(distribusi.index, distribusi.to_numpy(), color=["#9f1239", "#0f766e"])
        ax.bar_label(batang, padding=3)
        ax.set_title("Distribusi sentimen")
        ax.set_ylabel("Jumlah ulasan")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with kanan:
        rating = data[KOLOM_STARS].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        batang = ax.bar(rating.index.astype(str), rating.to_numpy(), color="#0f766e")
        ax.bar_label(batang, padding=3)
        ax.set_title("Distribusi rating")
        ax.set_xlabel("Bintang")
        ax.set_ylabel("Jumlah ulasan")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Istilah dengan rata-rata TF-IDF tertinggi pada data latih")
    if kata is None or kata.empty:
        _peringatan_berkas("hasil/kata_penting_tfidf.csv")
        return
    kolom_neg, kolom_pos = st.columns(2)
    for wadah, label, warna in (
        (kolom_neg, "negatif", "#9f1239"),
        (kolom_pos, "positif", "#0f766e"),
    ):
        bagian = kata.loc[kata["label"].eq(label)].sort_values("rata_tfidf")
        with wadah:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(bagian["kata"], bagian["rata_tfidf"], color=warna)
            ax.set_title(f"Kelas {label}")
            ax.set_xlabel("Rata-rata TF-IDF")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def halaman_prediksi() -> None:
    st.title("Prediksi sentimen")
    st.markdown(
        "Teks yang Anda tulis diproses dengan tahapan yang sama seperti data latih, "
        "lalu diklasifikasikan oleh model SVM yang sudah disimpan."
    )
    if not PATH_MODEL.exists() or not PATH_TFIDF.exists():
        _peringatan_berkas("models/svm_sentimen.joblib")
        return

    with st.form("form-prediksi"):
        ulasan = st.text_area(
            "Tulis ulasan",
            height=160,
            placeholder="Contoh: Tempatnya bersih, tetapi airnya kurang hangat.",
        )
        kirim = st.form_submit_button("Prediksi", type="primary")
    if not kirim:
        return
    if not ulasan or not ulasan.strip():
        st.info("Tuliskan ulasan terlebih dahulu.")
        return

    bersih = preprocess_teks(ulasan)
    if not bersih.strip():
        st.warning(
            "Setelah preprocessing tidak ada kata yang tersisa, "
            "jadi model tidak diminta memberi prediksi."
        )
        return

    model, vectorizer = _muat_model(PATH_MODEL.stat().st_mtime, PATH_TFIDF.stat().st_mtime)
    vektor = vectorizer.transform([bersih])
    prediksi = str(model.predict(vektor)[0])
    skor = float(model.decision_function(vektor)[0])
    kelas_skor_positif = str(model.classes_[1])
    warna = "#0f766e" if prediksi == "positif" else "#9f1239"
    st.markdown(
        f"<div class='kotak-hasil' style='background:{warna}'>Prediksi: {prediksi.capitalize()}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Skor keputusan SVM: {skor:.4f}. "
        f"Nilai di atas nol mengarah ke kelas {kelas_skor_positif}. "
        "Ini jarak dari garis pemisah, bukan persentase keyakinan."
    )
    st.text_area("Teks yang diterima model", bersih, height=120)


def halaman_evaluasi() -> None:
    st.title("Evaluasi model")
    metrik = _muat_json(PATH_METRIK)
    pelatihan = _muat_json(PATH_RINGKASAN_LATIH)
    if metrik is None:
        _peringatan_berkas("hasil/metrik_evaluasi.json")
        return

    if pelatihan:
        st.caption(
            f"Dievaluasi pada {pelatihan['jumlah_uji']} ulasan uji "
            f"({pelatihan['negatif_uji']} negatif, {pelatihan['positif_uji']} positif). "
            "TF-IDF hanya di-fit pada data latih. "
            f"SVM kernel {pelatihan['kernel_svm']}, tanpa class_weight."
        )

    atas = st.columns(2)
    bawah = st.columns(2)
    atas[0].metric("Accuracy", f"{metrik['accuracy']:.4f}")
    atas[1].metric("Precision macro", f"{metrik['precision_macro']:.4f}")
    bawah[0].metric("Recall macro", f"{metrik['recall_macro']:.4f}")
    bawah[1].metric("F1-score macro", f"{metrik['f1_macro']:.4f}")
    st.caption(
        "Precision, recall, dan F1 di atas adalah rata-rata macro, "
        "yaitu kedua kelas dihitung setara. Ini penting karena ulasan positif jauh lebih banyak. "
        f"Versi weighted: precision {metrik['precision_weighted']:.4f}, "
        f"recall {metrik['recall_weighted']:.4f}, "
        f"F1 {metrik['f1_weighted']:.4f}."
    )

    laporan = metrik["classification_report"]
    tabel = []
    for label in [*LABEL_URUTAN, "macro avg", "weighted avg"]:
        if label not in laporan:
            continue
        isi = laporan[label]
        tabel.append(
            {
                "Kelas": label,
                "Precision": isi["precision"],
                "Recall": isi["recall"],
                "F1-score": isi["f1-score"],
                "Support": int(isi["support"]),
            }
        )
    st.subheader("Classification report")
    st.dataframe(pd.DataFrame(tabel), use_container_width=True, hide_index=True)
    if PATH_LAPORAN_KELAS.exists():
        with st.expander("Teks classification report dari scikit-learn"):
            st.code(PATH_LAPORAN_KELAS.read_text(encoding="utf-8"))

    st.subheader("Confusion matrix")
    st.caption("Baris adalah label sebenarnya. Kolom adalah prediksi model.")
    matriks = metrik["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        matriks,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=metrik["label_urutan"],
        yticklabels=metrik["label_urutan"],
        ax=ax,
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    prediksi = _muat_csv(PATH_UJI_PREDIKSI)
    if prediksi is None:
        return
    salah = prediksi.loc[prediksi[KOLOM_LABEL].astype(str) != prediksi["prediksi"].astype(str)]
    st.subheader("Ulasan uji yang salah diklasifikasikan")
    st.caption(f"{len(salah)} dari {len(prediksi)} ulasan uji.")
    if salah.empty:
        st.info("Tidak ada ulasan uji yang salah diklasifikasikan.")
        return
    st.dataframe(
        salah[[KOLOM_STARS, KOLOM_LABEL, "prediksi", KOLOM_TEKS]],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    _terapkan_gaya()
    st.sidebar.title("Sari Ater")
    st.sidebar.caption("Analisis sentimen ulasan Google Maps")
    halaman = st.sidebar.radio(
        "Halaman",
        [
            "Dashboard",
            "Data Ulasan",
            "Preprocessing",
            "Analisis Sentimen",
            "Prediksi",
            "Evaluasi Model",
        ],
        label_visibility="collapsed",
    )
    if halaman == "Dashboard":
        halaman_dashboard()
    elif halaman == "Data Ulasan":
        halaman_data()
    elif halaman == "Preprocessing":
        halaman_preprocessing()
    elif halaman == "Analisis Sentimen":
        halaman_analisis()
    elif halaman == "Prediksi":
        halaman_prediksi()
    else:
        halaman_evaluasi()


if __name__ == "__main__":
    main()
