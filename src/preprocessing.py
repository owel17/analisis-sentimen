"""
Tahap 3 — Text preprocessing.

Tahapan pada setiap ulasan:
1. Cleansing
2. Case folding
3. Tokenizing
4. Stopword removal, dengan kata negasi yang tetap disimpan
5. Stemming Sastrawi, tanpa mengubah kata negasi

Teks asli tetap ada di kolom text.
Hasil akhir disimpan di kolom teks_preprocessing.
"""

import re

import pandas as pd

from konfigurasi import (
    KATA_NEGASI,
    KOLOM_BERSIH,
    KOLOM_LABEL,
    KOLOM_TEKS,
    PATH_BERLABEL,
    PATH_CONTOH_PREPROCESSING,
    PATH_PREPROCESSING,
)

_STEMMER = None
_STOPWORD = None


def _siapkan_alat():
    """Siapkan stemmer dan daftar stopword sekali saja."""
    global _STEMMER, _STOPWORD
    if _STEMMER is not None and _STOPWORD is not None:
        return _STEMMER, _STOPWORD

    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    _STEMMER = StemmerFactory().create_stemmer()
    daftar = set(StopWordRemoverFactory().get_stop_words())
    daftar -= set(KATA_NEGASI)
    _STOPWORD = daftar
    return _STEMMER, _STOPWORD


def cleansing(teks: str) -> str:
    """Hapus URL, mention, hashtag, angka, dan karakter selain huruf."""
    teks = "" if teks is None else str(teks)
    teks = re.sub(r"https?://\S+|www\.\S+", " ", teks, flags=re.IGNORECASE)
    teks = re.sub(r"<[^>]+>", " ", teks)
    teks = re.sub(r"&[a-zA-Z]+;", " ", teks)
    teks = re.sub(r"[@#]\w+", " ", teks)
    teks = re.sub(r"[^A-Za-z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


def preprocess_teks(teks: str) -> str:
    """Jalankan seluruh tahapan preprocessing pada satu ulasan."""
    stemmer, stopword = _siapkan_alat()
    dibersihkan = cleansing(teks)
    huruf_kecil = dibersihkan.lower()
    token = huruf_kecil.split()
    tanpa_stopword = [kata for kata in token if kata not in stopword]
    dasar = []
    for kata in tanpa_stopword:
        if kata in KATA_NEGASI:
            dasar.append(kata)
        else:
            dasar.append(stemmer.stem(kata))
    return " ".join(kata for kata in dasar if kata)


def main() -> None:
    if not PATH_BERLABEL.exists():
        raise FileNotFoundError(
            "File data berlabel belum ada. Jalankan dulu: python src/labeling.py"
        )

    data = pd.read_csv(PATH_BERLABEL, encoding="utf-8-sig")
    if KOLOM_TEKS not in data.columns:
        raise KeyError(f"Kolom {KOLOM_TEKS} tidak ditemukan di data berlabel.")

    _siapkan_alat()
    data[KOLOM_BERSIH] = data[KOLOM_TEKS].map(preprocess_teks)
    kosong = data[KOLOM_BERSIH].str.strip().eq("")

    PATH_PREPROCESSING.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PATH_PREPROCESSING, index=False, encoding="utf-8-sig")

    uji_negasi = preprocess_teks(
        "Tidak bagus, airnya kurang bersih dan belum layak! https://contoh.com"
    )
    token_uji = set(uji_negasi.split())
    # Kalimat uji sengaja tidak memuat "bukan"; tiga kata lain wajib tersisa.
    wajib_tersisa = {"tidak", "kurang", "belum"}
    if not wajib_tersisa.issubset(token_uji):
        raise RuntimeError(
            "Kata negasi hilang pada kalimat uji. "
            f"Hasil: {uji_negasi}. Hilang: {sorted(wajib_tersisa - token_uji)}"
        )

    isi = [
        "CONTOH PREPROCESSING",
        "Contoh di bawah memakai kalimat buatan, bukan teks ulasan pengguna.",
        f"Jumlah baris: {len(data)}",
        f"Baris yang menjadi kosong setelah preprocessing: {int(kosong.sum())}",
        "",
        "Uji fungsi pada kalimat buatan (bukan data penelitian):",
        "  Asli : Tidak bagus, airnya kurang bersih dan belum layak! https://contoh.com",
        f"  Hasil: {uji_negasi}",
        "  Kata negasi pada hasil uji: " + ", ".join(sorted(wajib_tersisa & token_uji)),
    ]

    laporan = "\n".join(isi) + "\n"
    PATH_CONTOH_PREPROCESSING.parent.mkdir(parents=True, exist_ok=True)
    PATH_CONTOH_PREPROCESSING.write_text(laporan, encoding="utf-8")
    print(laporan)
    print(f"Dataset preprocessing disimpan di: {PATH_PREPROCESSING}")


if __name__ == "__main__":
    main()
