"""
Tahap 2 — Seleksi ulasan dan pelabelan sentimen.

Membaca CSV mentah, lalu menulis file baru.
File mentah tidak diubah dan tidak ada baris yang dihapus dari file itu.

Aturan:
- Hanya ulasan dengan originalLanguage = id.
- Hanya baris yang kolom text-nya terisi.
- Teks yang sama persis disisakan satu (kemunculan pertama).
- Bintang 1 dan 2 = negatif, bintang 4 dan 5 = positif.
- Bintang 3 tidak masuk klasifikasi biner.
"""

import json

import pandas as pd

from konfigurasi import (
    KOLOM_LABEL,
    KOLOM_STARS,
    KOLOM_TEKS,
    PATH_BERLABEL,
    PATH_MENTAH,
    PATH_RINGKASAN_DATA,
)


def teks_kosong(seri: pd.Series) -> pd.Series:
    """True jika sel kosong, NaN, atau hanya berisi spasi."""
    sebagai_teks = seri.astype("string")
    return sebagai_teks.isna() | sebagai_teks.str.strip().eq("")


def beri_label(nilai_bintang) -> str | None:
    """Ubah bintang menjadi label sentimen. Bintang 3 tidak diberi label."""
    if pd.isna(nilai_bintang):
        return None
    bintang = int(nilai_bintang)
    if bintang in (1, 2):
        return "negatif"
    if bintang in (4, 5):
        return "positif"
    return None


def main() -> None:
    if not PATH_MENTAH.exists():
        raise FileNotFoundError(f"File CSV mentah tidak ditemukan: {PATH_MENTAH}")

    mentah = pd.read_csv(PATH_MENTAH, low_memory=False)
    wajib = ["originalLanguage", KOLOM_TEKS, KOLOM_STARS, "publishedAtDate"]
    hilang = [nama for nama in wajib if nama not in mentah.columns]
    if hilang:
        raise KeyError(f"Kolom yang dibutuhkan tidak ada: {hilang}")

    punya_teks = ~teks_kosong(mentah[KOLOM_TEKS])
    bahasa_id = mentah["originalLanguage"].astype("string").str.strip().str.lower().eq("id")
    kandidat = mentah.loc[bahasa_id & punya_teks].copy()
    kandidat[KOLOM_STARS] = pd.to_numeric(kandidat[KOLOM_STARS], errors="coerce")
    kandidat["_teks"] = kandidat[KOLOM_TEKS].astype(str).str.strip()

    jumlah_konflik = int((kandidat.groupby("_teks")[KOLOM_STARS].nunique() > 1).sum())
    jumlah_sebelum_unik = len(kandidat)
    unik = kandidat.drop_duplicates(subset=["_teks"], keep="first").copy()

    unik[KOLOM_LABEL] = unik[KOLOM_STARS].map(beri_label)
    netral = unik[KOLOM_LABEL].isna()
    berlabel = unik.loc[~netral].copy()
    berlabel[KOLOM_STARS] = berlabel[KOLOM_STARS].astype(int)
    berlabel[KOLOM_TEKS] = berlabel["_teks"]

    kolom_simpan = [
        "publishedAtDate",
        "originalLanguage",
        KOLOM_STARS,
        KOLOM_TEKS,
        KOLOM_LABEL,
    ]
    berlabel = berlabel[kolom_simpan].reset_index(drop=True)

    PATH_BERLABEL.parent.mkdir(parents=True, exist_ok=True)
    berlabel.to_csv(PATH_BERLABEL, index=False, encoding="utf-8-sig")

    tanggal = pd.to_datetime(berlabel["publishedAtDate"], errors="coerce")
    hitung_label = berlabel[KOLOM_LABEL].value_counts()
    ringkasan = {
        "jumlah_baris_mentah": int(len(mentah)),
        "baris_tanpa_teks": int((~punya_teks).sum()),
        "baris_dengan_teks": int(punya_teks.sum()),
        "bahasa_indonesia_dan_punya_teks": int((bahasa_id & punya_teks).sum()),
        "duplikat_teks_diabaikan": int(jumlah_sebelum_unik - len(unik)),
        "teks_sama_dengan_bintang_berbeda": jumlah_konflik,
        "setelah_duplikat": int(len(unik)),
        "rating_3_dikeluarkan": int(netral.sum()),
        "data_klasifikasi": int(len(berlabel)),
        "positif": int(hitung_label.get("positif", 0)),
        "negatif": int(hitung_label.get("negatif", 0)),
        "tanggal_paling_awal": None if tanggal.isna().all() else str(tanggal.min()),
        "tanggal_paling_akhir": None if tanggal.isna().all() else str(tanggal.max()),
        "file_hasil": PATH_BERLABEL.name,
    }
    PATH_RINGKASAN_DATA.parent.mkdir(parents=True, exist_ok=True)
    PATH_RINGKASAN_DATA.write_text(
        json.dumps(ringkasan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Tahap 2 selesai. Dataset tersanitasi tidak diubah.")
    print(json.dumps(ringkasan, indent=2, ensure_ascii=False))
    if jumlah_konflik:
        print(
            "Perhatian: ada teks yang sama tetapi bintangnya berbeda. "
            "Yang disimpan adalah kemunculan pertama."
        )


if __name__ == "__main__":
    main()
