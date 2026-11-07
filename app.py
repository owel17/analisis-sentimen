"""
Dashboard Streamlit untuk hasil penelitian analisis sentimen Sari Ater
dan ulasan pengunjung per kolam.

Halaman penelitian hanya membaca data, model, dan metrik yang sudah dihasilkan
script di folder src. Ulasan baru disimpan terpisah dan tidak melatih ulang model.
"""

import hmac
import html
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

plt.rcParams.update(
    {
        "figure.facecolor": "#fffdf8",
        "axes.facecolor": "#fffdf8",
        "savefig.facecolor": "#fffdf8",
        "axes.edgecolor": "#e4dccb",
        "text.color": "#142826",
        "axes.labelcolor": "#142826",
        "xtick.color": "#142826",
        "ytick.color": "#142826",
        "font.family": "sans-serif",
    }
)

PITA_KOLAM = {
    "Kolam Mayangsari": "linear-gradient(128deg, #0f766e 0%, #115e59 58%, #c2410c 140%)",
    "Kolam Wangsadipa": "linear-gradient(128deg, #0e7490 0%, #155e75 55%, #0369a1 100%)",
    "Kolam Pulosari": "linear-gradient(145deg, #14532d 0%, #0f766e 52%, #65a30d 125%)",
    "Kolam Leuwisari": "linear-gradient(128deg, #c2410c 0%, #9a3412 48%, #0f766e 130%)",
    "Sariater Waterpark": "linear-gradient(128deg, #0369a1 0%, #0e7490 46%, #22d3ee 130%)",
}

AKAR = Path(__file__).resolve().parent
PATH_LOGO = AKAR / "assets" / "logo-sari-ater.png"
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
from ulasan_pengguna import (  # noqa: E402
    DAFTAR_KOLAM,
    KOLOM_BINTANG,
    KOLOM_KOLAM,
    KOLOM_LABEL_ULASAN,
    KOLOM_NAMA,
    KOLOM_TEKS_ULASAN,
    KOLOM_WAKTU,
    muat_ulasan,
    nama_tampilan,
    tambah_ulasan,
)


st.set_page_config(
    page_title="Sentimen Sari Ater",
    page_icon=str(PATH_LOGO),
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
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,680&family=Manrope:wght@400;500;600;700;800&display=swap');
        .stApp {
            background:
                radial-gradient(980px 420px at 100% -8%, rgba(234, 148, 42, 0.22), transparent 58%),
                radial-gradient(820px 460px at -12% 8%, rgba(14, 148, 136, 0.18), transparent 52%),
                radial-gradient(640px 360px at 70% 100%, rgba(3, 105, 161, 0.08), transparent 50%),
                linear-gradient(180deg, #f7f1e4 0%, #eef6f3 42%, #f6f3ea 100%);
            color: #142826;
        }
        html, body, [class*="css"] { font-family: Manrope, sans-serif; }
        h1, h2, h3, .judul-aplikasi, .judul-halaman, .nama-merek, .nama-di-pita {
            font-family: Fraunces, Georgia, serif !important;
            letter-spacing: -0.03em;
        }
        header[data-testid="stHeader"] { background: transparent; }
        h2 {
            color: #143d3a;
            border-left: 4px solid #e09a2b;
            padding-left: 0.65rem;
            margin-top: 0.4rem;
        }
        .block-container { padding-top: 1.4rem; max-width: 1180px; }
        section[data-testid="stSidebar"] {
            background:
                radial-gradient(420px 220px at 20% 0%, rgba(224, 154, 43, 0.28), transparent 60%),
                linear-gradient(180deg, #0e3330 0%, #143d3a 42%, #0c2f38 100%);
            border-right: 0;
        }
        section[data-testid="stSidebar"] * { color: #f7f5f0; }
        section[data-testid="stSidebar"] a { color: #f7f5f0; }
        section[data-testid="stSidebar"] [data-testid="stImage"] img {
            background: rgba(255, 253, 248, 0.12);
            border-radius: 16px;
            padding: 4px;
            width: 58px;
            height: 58px;
            object-fit: contain;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #d7efe9;
            letter-spacing: 0.01em;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 12px;
            padding: 0.28rem 0.45rem;
            margin-bottom: 0.12rem;
            transition: background 0.15s ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.08);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: rgba(224, 154, 43, 0.28);
        }
        .nama-merek {
            font-size: 1.7rem;
            font-weight: 680;
            line-height: 1.05;
            margin: 0;
            color: #fffaf2;
        }
        .hero, .kepala {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 1.7rem 1.8rem 1.55rem;
            margin: 0 0 1.15rem 0;
            color: #fffaf2;
            box-shadow: 0 18px 40px rgba(15, 61, 56, 0.16);
        }
        .hero {
            background: linear-gradient(120deg, #0c3330 0%, #0f766e 38%, #b45309 72%, #0e7490 100%);
            background-size: 180% 180%;
            animation: uap 18s ease infinite;
            min-height: 168px;
        }
        .kepala-riset {
            background: linear-gradient(120deg, #123f44 0%, #0f766e 62%, #1d4e89 130%);
        }
        .kepala-tamu {
            background: linear-gradient(120deg, #7c2d12 0%, #c2410c 46%, #0f766e 120%);
        }
        .hero::after, .kepala::after {
            content: "";
            position: absolute;
            right: -40px;
            bottom: -46px;
            width: 220px;
            height: 120px;
            background:
                radial-gradient(60px 28px at 30% 60%, rgba(255,255,255,0.18), transparent 70%),
                radial-gradient(80px 34px at 70% 50%, rgba(255,255,255,0.12), transparent 72%);
            pointer-events: none;
        }
        .kicker {
            margin: 0 0 0.35rem 0;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #ffe7b8;
        }
        .judul-halaman, .judul-hero {
            color: #fffaf2;
            font-weight: 680;
            line-height: 1.08;
            margin: 0;
        }
        .judul-hero { font-size: 3rem; }
        .judul-halaman { font-size: 2.15rem; }
        .pengantar, .pengantar-kepala {
            color: rgba(255, 250, 242, 0.9);
            font-size: 1.05rem;
            margin: 0.55rem 0 0 0;
            max-width: 40rem;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #fffdf8 0%, #fff 100%);
            border: 1px solid #efe6d4;
            border-top: 4px solid #0f766e;
            border-radius: 18px;
            padding: 14px 16px 12px;
            box-shadow: 0 12px 28px rgba(20, 61, 58, 0.06);
        }
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] {
            border-top-color: #c2410c;
        }
        div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] {
            border-top-color: #0369a1;
        }
        div[data-testid="stMetricValue"] {
            font-family: Fraunces, Georgia, serif;
            color: #143d3a;
        }
        .stButton > button {
            border-radius: 999px;
            font-weight: 700;
            border: 1px solid #d9cbb3;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.16);
            border-color: #0f766e;
            color: #0f766e;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0f766e, #0e7490);
            border: none;
            color: white;
        }
        .stButton > button[kind="primary"]:hover { color: white; }
        div[data-testid="stDataFrame"] {
            border: 1px solid #efe6d4;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(20, 61, 58, 0.05);
            background: #fffdf8;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 22px !important;
            border: 1px solid #efe6d4 !important;
            background: rgba(255, 253, 248, 0.78);
            box-shadow: 0 10px 26px rgba(20, 61, 58, 0.05);
        }
        div[data-testid="stAlert"] { border-radius: 16px; }
        .kotak-hasil {
            border-radius: 22px;
            padding: 22px 24px;
            color: white;
            font-family: Fraunces, Georgia, serif;
            font-size: 1.7rem;
            font-weight: 680;
            box-shadow: 0 16px 32px rgba(20, 61, 58, 0.14);
        }
        div[data-testid="stHorizontalBlock"] { align-items: stretch; }
        div[data-testid="stColumn"]:has(.pita-kolam) { display: flex; }
        div[data-testid="stColumn"]:has(.pita-kolam) > div,
        div[data-testid="stColumn"]:has(.pita-kolam) [data-testid="stLayoutWrapper"],
        div[data-testid="stColumn"]:has(.pita-kolam) [data-testid="stVerticalBlock"] {
            flex: 1;
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.pita-kolam) {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            padding: 0 !important;
        }
        div[data-testid="stVerticalBlock"]:has(.pita-kolam):not(:has([data-testid="stVerticalBlock"])) {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #fffdf8;
            border: none !important;
            border-radius: 24px !important;
            padding: 0 0 1rem;
            overflow: hidden;
            box-shadow: 0 16px 36px rgba(20, 61, 58, 0.1);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        div[data-testid="stVerticalBlock"]:has(.pita-kolam):not(:has([data-testid="stVerticalBlock"])):hover {
            transform: translateY(-4px);
            box-shadow: 0 22px 44px rgba(20, 61, 58, 0.16);
        }
        div[data-testid="stVerticalBlock"]:has(.pita-kolam):not(:has([data-testid="stVerticalBlock"])) .stButton {
            margin-top: auto;
            padding: 0 1.05rem;
        }
        div[data-testid="stVerticalBlock"]:has(.pita-kolam):not(:has([data-testid="stVerticalBlock"])) [data-testid="stElementContainer"]:has(.stButton) {
            margin-top: auto;
        }
        .pita-kolam {
            position: relative;
            min-height: 118px;
            margin: 0 0 0.95rem 0;
            padding: 1.15rem 1.15rem 1rem;
            overflow: hidden;
        }
        .pita-kolam::after {
            content: "";
            position: absolute;
            right: -20px;
            bottom: -28px;
            width: 140px;
            height: 70px;
            border-radius: 80px 80px 0 0;
            background: rgba(255, 255, 255, 0.14);
        }
        .nama-di-pita {
            position: relative;
            z-index: 1;
            color: #fffaf2;
            font-size: 1.28rem;
            font-weight: 680;
            line-height: 1.2;
            margin: 0;
            max-width: 14rem;
        }
        .isi-kartu { padding: 0 1.15rem; }
        .angka-bintang {
            color: #143d3a;
            font-family: Fraunces, Georgia, serif;
            font-size: 2.5rem;
            font-weight: 680;
            line-height: 1;
            margin: 0.1rem 0;
        }
        .bintang-visual { color: #d97706; letter-spacing: 0.08em; font-size: 1.15rem; white-space: nowrap; }
        .meta-kolam { color: #5c6b69; margin: 0.2rem 0 0.85rem 0; }
        .kartu-ulasan {
            background: #fffdf8;
            border: 1px solid #efe6d4;
            border-left: 5px solid #0f766e;
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: 0 8px 20px rgba(20, 61, 58, 0.05);
        }
        .kartu-ulasan:has(.lencana-negatif) { border-left-color: #be123c; }
        .baris-ulasan {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem 0.9rem;
            align-items: baseline;
            margin-bottom: 0.35rem;
        }
        .nama-ulasan { color: #143d3a; font-weight: 800; }
        .waktu-ulasan { color: #6b7280; font-size: 0.92rem; }
        .teks-ulasan { color: #142826; margin: 0.35rem 0 0.55rem 0; white-space: pre-wrap; }
        .lencana {
            display: inline-block;
            border-radius: 999px;
            padding: 0.18rem 0.7rem;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }
        .lencana-positif { background: #ccfbf1; color: #115e59; }
        .lencana-negatif { background: #ffe4e6; color: #9f1239; }
        div[data-testid="stCustomComponentV1"] {
            height: 0;
            overflow: hidden;
        }
        @media (prefers-reduced-motion: reduce) {
            .hero { animation: none; }
            div[data-testid="stVerticalBlock"]:has(.pita-kolam):not(:has([data-testid="stHorizontalBlock"])),
            .stButton > button { transition: none; }
        }
        @keyframes uap {
            0% { background-position: 0% 40%; }
            50% { background-position: 100% 60%; }
            100% { background-position: 0% 40%; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kepala(judul: str, pengantar: str = "", ragam: str = "riset") -> None:
    kelas = "kepala-tamu" if ragam == "tamu" else "kepala-riset"
    kicker = "Pengunjung" if ragam == "tamu" else "Penelitian"
    paragraf = (
        f"<p class='pengantar-kepala'>{html.escape(pengantar)}</p>" if pengantar else ""
    )
    st.markdown(
        f"<div class='kepala {kelas}'>"
        f"<p class='kicker'>{kicker}</p>"
        f"<h1 class='judul-halaman'>{html.escape(judul)}</h1>"
        f"{paragraf}</div>",
        unsafe_allow_html=True,
    )


def _peringatan_berkas(nama: str) -> None:
    st.warning(
        f"Berkas `{nama}` belum ada. Jalankan dulu tahapan penelitian dari terminal, "
        "lalu muat ulang halaman ini."
    )


def halaman_dashboard() -> None:
    _kepala(
        "Dashboard",
        "Ringkasan data ulasan Pemandian Air Panas Sari Ater yang dipakai untuk klasifikasi sentimen.",
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
    _kepala("Data ulasan", "Ulasan berlabel yang menjadi bahan klasifikasi sentimen.")
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
    _kepala(
        "Preprocessing",
        "Perbandingan teks asli dan hasil cleansing, case folding, tokenizing, "
        "stopword removal, serta stemming. Kata tidak, bukan, belum, dan kurang tidak dihapus.",
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
    _kepala("Analisis sentimen", "Sebaran sentimen, rating, dan istilah yang paling menonjol.")
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
    _kepala(
        "Prediksi sentimen",
        "Teks yang Anda tulis diproses dengan tahapan yang sama seperti data latih, "
        "lalu diklasifikasikan oleh model SVM yang sudah disimpan.",
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
    _kepala("Evaluasi model", "Metrik pada data uji. Model tidak dilatih ulang di halaman ini.")
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


def _model_siap() -> bool:
    return PATH_MODEL.exists() and PATH_TFIDF.exists()


def _prediksi_label(teks_bersih: str) -> str:
    model, vectorizer = _muat_model(PATH_MODEL.stat().st_mtime, PATH_TFIDF.stat().st_mtime)
    vektor = vectorizer.transform([teks_bersih])
    return str(model.predict(vektor)[0])


def _rahasia_admin() -> tuple[str, str] | None:
    try:
        admin = st.secrets["admin"]
        nama = str(admin["nama"]).strip()
        sandi = str(admin["kata_sandi"])
    except (KeyError, StreamlitSecretNotFoundError, AttributeError):
        return None
    if not nama or not sandi:
        return None
    return nama, sandi


def _pasang_menu_login() -> None:
    """Tambahkan tautan Login admin ke menu titik tiga Streamlit."""
    import streamlit.components.v1 as components

    components.html(
        """
        <script>
        const dokumen = window.parent.document;
        function pasang() {
          const daftar = dokumen.querySelector('[data-testid="stMainMenuList"]');
          if (!daftar || dokumen.getElementById("menu-login-admin")) return;
          const contoh = daftar.querySelector('[data-testid="stMainMenuItem-rerun"]');
          if (!contoh) return;
          const tombol = dokumen.createElement("a");
          tombol.id = "menu-login-admin";
          tombol.href = window.parent.location.pathname + "?halaman=login";
          tombol.setAttribute("role", "menuitem");
          tombol.className = contoh.className;
          tombol.style.display = "flex";
          tombol.style.alignItems = "center";
          tombol.style.textDecoration = "none";
          tombol.style.color = "inherit";
          tombol.textContent = "Login admin";
          daftar.insertBefore(tombol, daftar.firstChild);
        }
        if (!dokumen.body.dataset.pemantauLogin) {
          dokumen.body.dataset.pemantauLogin = "ya";
          dokumen.addEventListener("click", function (kejadian) {
            const target = kejadian.target;
            if (!target || target.id !== "menu-login-admin") return;
            kejadian.stopPropagation();
          }, true);
        }
        pasang();
        new MutationObserver(pasang).observe(dokumen.body, {childList: true, subtree: true});
        </script>
        """,
        height=0,
        width=0,
    )


def halaman_login() -> None:
    _kepala(
        "Login admin",
        "Masuk untuk melihat ringkasan seluruh kolam dan hasil penelitian. "
        "Pengunjung tidak perlu login untuk memberi ulasan.",
        ragam="tamu",
    )
    if st.button("Kembali ke kolam"):
        if "halaman" in st.query_params:
            del st.query_params["halaman"]
        st.rerun()

    rahasia = _rahasia_admin()
    if rahasia is None:
        st.warning(
            "Login admin belum disiapkan. Salin `.streamlit/secrets.toml.example` "
            "menjadi `.streamlit/secrets.toml`, lalu isi nama dan kata sandi."
        )
        return

    with st.form("login-admin"):
        nama = st.text_input("Nama")
        sandi = st.text_input("Kata sandi", type="password")
        masuk = st.form_submit_button("Masuk", type="primary")
    if not masuk:
        return

    nama_benar, sandi_benar = rahasia
    nama_cocok = hmac.compare_digest(nama.strip().encode(), nama_benar.encode())
    sandi_cocok = hmac.compare_digest(sandi.encode(), sandi_benar.encode())
    if nama_cocok and sandi_cocok:
        st.session_state.admin = True
        st.session_state.halaman = "Ringkasan Kolam"
        if "halaman" in st.query_params:
            del st.query_params["halaman"]
        st.rerun()
    st.error("Nama atau kata sandi tidak sesuai.")


def _ringkas_kolam(data: pd.DataFrame, kolam: str) -> dict:
    bagian = data.loc[data[KOLOM_KOLAM].astype(str).eq(kolam)]
    jumlah = int(len(bagian))
    if jumlah == 0:
        return {"jumlah": 0, "rata": None, "positif": 0, "negatif": 0}
    return {
        "jumlah": jumlah,
        "rata": float(bagian[KOLOM_BINTANG].mean()),
        "positif": int(bagian[KOLOM_LABEL_ULASAN].astype(str).eq("positif").sum()),
        "negatif": int(bagian[KOLOM_LABEL_ULASAN].astype(str).eq("negatif").sum()),
    }


def _format_rata(rata: float | None) -> str:
    if rata is None:
        return "—"
    return f"{rata:.1f}".replace(".", ",")


def _bintang_visual(nilai: float | None) -> str:
    if nilai is None:
        return "☆☆☆☆☆"
    penuh = max(0, min(5, int(round(float(nilai)))))
    return "★" * penuh + "☆" * (5 - penuh)


def _kartu_kolam(kolam: str, ringkas: dict) -> None:
    warna = PITA_KOLAM.get(kolam, PITA_KOLAM["Kolam Mayangsari"])
    if ringkas["jumlah"] == 0:
        isi = "<p class='meta-kolam'>Belum ada ulasan</p>"
    else:
        isi = (
            f"<p class='angka-bintang'>{_format_rata(ringkas['rata'])}</p>"
            f"<p class='bintang-visual'>{_bintang_visual(ringkas['rata'])}</p>"
            f"<p class='meta-kolam'>{ringkas['jumlah']} ulasan</p>"
        )
    st.markdown(
        f"<div class='pita-kolam' style='background:{warna}'>"
        f"<p class='nama-di-pita'>{html.escape(kolam)}</p>"
        f"</div>"
        f"<div class='isi-kartu'>{isi}</div>",
        unsafe_allow_html=True,
    )


def _kartu_ulasan(baris: pd.Series) -> None:
    label = str(baris[KOLOM_LABEL_ULASAN])
    kelas = "lencana-positif" if label == "positif" else "lencana-negatif"
    bintang = int(baris[KOLOM_BINTANG]) if pd.notna(baris[KOLOM_BINTANG]) else None
    st.markdown(
        "<div class='kartu-ulasan'>"
        "<div class='baris-ulasan'>"
        f"<span class='nama-ulasan'>{html.escape(nama_tampilan(baris[KOLOM_NAMA]))}</span>"
        f"<span class='bintang-visual'>{_bintang_visual(bintang)}</span>"
        f"<span class='waktu-ulasan'>{html.escape(str(baris[KOLOM_WAKTU]))}</span>"
        "</div>"
        f"<p class='teks-ulasan'>{html.escape(str(baris[KOLOM_TEKS_ULASAN]))}</p>"
        f"<span class='lencana {kelas}'>{html.escape(label)}</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def halaman_kolam() -> None:
    if st.session_state.get("kolam_dipilih") in DAFTAR_KOLAM:
        _halaman_detail_kolam(st.session_state.kolam_dipilih)
        return

    st.markdown(
        "<div class='hero'>"
        "<p class='kicker'>Pemandian Air Panas</p>"
        "<h1 class='judul-hero'>Sari Ater</h1>"
        "<p class='pengantar'>Pilih kolam, baca ulasan pengunjung, "
        "lalu beri bintang dan ulasan Anda.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    data = muat_ulasan()
    kolom = [*st.columns(3), *st.columns(2)]
    for wadah, kolam in zip(kolom, DAFTAR_KOLAM):
        ringkas = _ringkas_kolam(data, kolam)
        with wadah:
            with st.container(border=True):
                _kartu_kolam(kolam, ringkas)
                if st.button("Buka", key=f"buka-{kolam}", use_container_width=True):
                    st.session_state.kolam_dipilih = kolam
                    st.rerun()


def _halaman_detail_kolam(kolam: str) -> None:
    if st.session_state.pop("bersihkan_form", False):
        st.session_state["nomor_form"] = st.session_state.get("nomor_form", 0) + 1

    if st.button("Kembali ke daftar kolam"):
        st.session_state.kolam_dipilih = None
        st.rerun()

    st.markdown(
        f"<div class='kepala' style='background:{PITA_KOLAM.get(kolam, PITA_KOLAM['Kolam Mayangsari'])}'>"
        f"<p class='kicker'>Kolam</p>"
        f"<h1 class='judul-halaman'>{html.escape(kolam)}</h1>"
        "</div>",
        unsafe_allow_html=True,
    )
    pesan = st.session_state.pop("pesan_ulasan", None)
    if pesan:
        st.success(pesan)

    data = muat_ulasan()
    bagian = data.loc[data[KOLOM_KOLAM].astype(str).eq(kolam)].copy()
    ringkas = _ringkas_kolam(data, kolam)
    if ringkas["jumlah"] == 0:
        st.info("Belum ada ulasan untuk kolam ini.")
    else:
        kiri, kanan = st.columns(2)
        kiri.metric("Rata-rata bintang", _format_rata(ringkas["rata"]))
        kanan.metric("Jumlah ulasan", ringkas["jumlah"])

    if ringkas["jumlah"] > 0:
        st.markdown(
            f"<p class='bintang-visual'>{_bintang_visual(ringkas['rata'])}</p>",
            unsafe_allow_html=True,
        )

    st.subheader("Beri ulasan")
    if not _model_siap():
        _peringatan_berkas("models/svm_sentimen.joblib")
    else:
        nomor_form = st.session_state.get("nomor_form", 0)
        pilihan_bintang = st.feedback("stars", key=f"bintang-{kolam}-{nomor_form}")
        with st.form(f"form-ulasan-{kolam}-{nomor_form}"):
            ulasan = st.text_area(
                "Ulasan",
                height=140,
                placeholder="Contoh: Airnya hangat dan kolamnya bersih.",
            )
            nama = st.text_input("Nama tampilan", placeholder="Boleh dikosongkan")
            kirim = st.form_submit_button("Kirim ulasan", type="primary")
        if kirim:
            if pilihan_bintang is None:
                st.warning("Pilih bintang terlebih dahulu.")
            else:
                _simpan_ulasan_kolam(kolam, int(pilihan_bintang) + 1, ulasan, nama)

    st.subheader("Ulasan pengunjung")
    if bagian.empty:
        return
    saringan = st.pills(
        "Saring menurut bintang",
        [5, 4, 3, 2, 1],
        selection_mode="multi",
        default=[5, 4, 3, 2, 1],
        format_func=lambda nilai: "★" * int(nilai),
        key=f"saring-{kolam}",
    )
    if not saringan:
        st.info("Pilih sedikitnya satu bintang untuk melihat ulasan.")
        return
    tampilan = bagian.loc[bagian[KOLOM_BINTANG].isin(saringan)].sort_values(
        KOLOM_WAKTU, ascending=False
    )
    if tampilan.empty:
        st.info("Tidak ada ulasan dengan bintang yang dipilih.")
        return
    for _, baris in tampilan.iterrows():
        _kartu_ulasan(baris)


def _simpan_ulasan_kolam(kolam: str, bintang: int, ulasan: str, nama: str) -> None:
    if not ulasan or not ulasan.strip():
        st.info("Tuliskan ulasan terlebih dahulu.")
        return
    if len(" ".join(nama.split())) > 40:
        st.warning("Nama tampilan paling panjang 40 karakter.")
        return

    bersih = preprocess_teks(ulasan)
    if not bersih.strip():
        st.warning(
            "Setelah preprocessing tidak ada kata yang tersisa, "
            "jadi ulasan tidak disimpan dan model tidak diminta memberi prediksi."
        )
        return

    label = _prediksi_label(bersih)
    tambah_ulasan(kolam, bintang, ulasan.strip(), label, nama)
    st.session_state.pesan_ulasan = (
        f"Ulasan tersimpan. Bintang {bintang} · prediksi sentimen {label}."
    )
    st.session_state.bersihkan_form = True
    st.rerun()


def halaman_ringkasan_kolam() -> None:
    _kepala(
        "Ringkasan kolam",
        "Gabungan ulasan yang dikirim pengunjung. "
        "Angka ini terpisah dari dataset Google Maps pada halaman penelitian.",
        ragam="tamu",
    )
    data = muat_ulasan()
    if data.empty:
        st.info("Belum ada ulasan pengunjung.")
        return

    positif = int(data[KOLOM_LABEL_ULASAN].astype(str).eq("positif").sum())
    negatif = int(data[KOLOM_LABEL_ULASAN].astype(str).eq("negatif").sum())
    atas = st.columns(3)
    atas[0].metric("Total ulasan baru", len(data))
    atas[1].metric("Prediksi positif", positif)
    atas[2].metric("Prediksi negatif", negatif)

    tabel = []
    jumlah_positif = []
    jumlah_negatif = []
    for kolam in DAFTAR_KOLAM:
        ringkas = _ringkas_kolam(data, kolam)
        jumlah_positif.append(ringkas["positif"])
        jumlah_negatif.append(ringkas["negatif"])
        tabel.append(
            {
                "Kolam": kolam,
                "Jumlah ulasan": ringkas["jumlah"],
                "Rata-rata bintang": _format_rata(ringkas["rata"]),
                "Positif": ringkas["positif"],
                "Negatif": ringkas["negatif"],
            }
        )
    st.subheader("Per kolam")
    st.dataframe(pd.DataFrame(tabel), use_container_width=True, hide_index=True)

    st.subheader("Prediksi sentimen per kolam")
    st.caption("Arahkan kursor ke batang untuk melihat jumlahnya.")
    import altair as alt

    nama_pendek = ["Mayangsari", "Wangsadipa", "Pulosari", "Leuwisari", "Waterpark"]
    sumber = pd.DataFrame(
        {
            "Kolam": nama_pendek * 2,
            "Sentimen": ["positif"] * 5 + ["negatif"] * 5,
            "Jumlah": jumlah_positif + jumlah_negatif,
        }
    )
    grafik = (
        alt.Chart(sumber)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Kolam:N", sort=nama_pendek, axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Jumlah:Q", axis=alt.Axis(title="Jumlah ulasan", tickMinStep=1)),
            color=alt.Color(
                "Sentimen:N",
                scale=alt.Scale(domain=["positif", "negatif"], range=["#0f766e", "#9f1239"]),
                legend=alt.Legend(title=None),
            ),
            xOffset="Sentimen:N",
            tooltip=["Kolam", "Sentimen", "Jumlah"],
        )
        .properties(background="#fffdf8", height=320)
    )
    st.altair_chart(grafik, use_container_width=True)


HALAMAN_PUBLIK = ["Kolam"]
HALAMAN_ADMIN = [
    "Kolam",
    "Ringkasan Kolam",
    "Dashboard",
    "Data Ulasan",
    "Preprocessing",
    "Analisis Sentimen",
    "Prediksi",
    "Evaluasi Model",
]


def main() -> None:
    _terapkan_gaya()
    st.session_state.setdefault("admin", False)
    admin = bool(st.session_state.admin)

    kolom_logo, kolom_nama = st.sidebar.columns([1, 3], vertical_alignment="center")
    kolom_logo.image(str(PATH_LOGO), width=58)
    kolom_nama.markdown("<p class='nama-merek'>Sari Ater</p>", unsafe_allow_html=True)
    if admin:
        st.sidebar.caption("Ulasan kolam dan hasil penelitian")
        if st.sidebar.button("Keluar", use_container_width=True):
            st.session_state.admin = False
            st.session_state.halaman = "Kolam"
            st.rerun()
    else:
        st.sidebar.caption("Ulasan kolam Sari Ater")
        _pasang_menu_login()

    if admin and st.query_params.get("halaman") == "login":
        del st.query_params["halaman"]
    if not admin and st.query_params.get("halaman") == "login":
        halaman_login()
        return

    pilihan = HALAMAN_ADMIN if admin else HALAMAN_PUBLIK
    if st.session_state.get("halaman") not in pilihan:
        st.session_state.halaman = "Kolam"
    halaman = st.sidebar.radio(
        "Halaman",
        pilihan,
        key="halaman",
        label_visibility="collapsed",
    )

    if halaman == "Kolam":
        halaman_kolam()
    elif halaman == "Ringkasan Kolam":
        halaman_ringkasan_kolam()
    elif halaman == "Dashboard":
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
