"""Audit dataset ulasan Sari Ater yang telah disanitasi."""

from pathlib import Path

import pandas as pd

# Folder proyek = satu tingkat di atas folder src/
AKAR_PROYEK = Path(__file__).resolve().parents[1]
NAMA_FILE = "dataset_Google-Maps-Reviews-Scraper_2026-11-04_16-44-52-438.csv"
PATH_CSV = AKAR_PROYEK / "dataset" / NAMA_FILE
PATH_LAPORAN = AKAR_PROYEK / "hasil" / "laporan_audit_dataset.txt"

# Kolom yang dipakai untuk memahami data penelitian.
KOLOM_KUNCI = [
    "text",
    "stars",
    "publishedAtDate",
    "originalLanguage",
]


def teks_kosong(seri: pd.Series) -> pd.Series:
    """True jika sel kosong, NaN, atau hanya spasi."""
    return seri.isna() | seri.astype(str).str.strip().eq("") | seri.astype(str).str.strip().str.lower().eq("nan")


def blok(judul: str) -> str:
    garis = "=" * 72
    return f"\n{garis}\n{judul}\n{garis}\n"


def format_hitungan(seri: pd.Series, nama_kosong: str = "(kosong)") -> str:
    """Ubah value_counts menjadi teks yang mudah dibaca."""
    sebagai_teks = seri.astype("string")
    hitung = sebagai_teks.fillna(nama_kosong).value_counts(dropna=False)
    baris = [f"  {nilai}: {jumlah}" for nilai, jumlah in hitung.items()]
    return "\n".join(baris)


def main() -> None:
    if not PATH_CSV.exists():
        raise FileNotFoundError(f"File CSV tidak ditemukan: {PATH_CSV}")

    # Baca dataset yang telah disanitasi; laporan tidak memuat teks ulasan.
    df = pd.read_csv(PATH_CSV, low_memory=False)

    bagian = []
    bagian.append("LAPORAN AUDIT DATASET")
    bagian.append("Analisis Sentimen Ulasan Google Maps — Pemandian Air Panas Sari Ater")
    bagian.append(f"File: {PATH_CSV.name}")
    bagian.append("Catatan: dataset yang digunakan telah disanitasi untuk menghapus identitas reviewer.")

    bagian.append(blok("1. Ukuran data"))
    bagian.append(f"Jumlah baris: {len(df)}")
    bagian.append(f"Jumlah kolom: {len(df.columns)}")

    bagian.append(blok("2. Nama kolom"))
    for indeks, nama in enumerate(df.columns, start=1):
        bagian.append(f"{indeks:3}. {nama}")

    bagian.append(blok("3. Kolom kunci yang diharapkan"))
    for nama in KOLOM_KUNCI:
        status = "ada" if nama in df.columns else "TIDAK ADA"
        bagian.append(f"  {nama}: {status}")

    bagian.append(blok("4. Data kosong"))
    kosong = df.isna().sum()
    bagian.append(f"Total sel kosong (NaN): {int(kosong.sum())}")
    bagian.append(f"Kolom yang seluruhnya kosong: {int((kosong == len(df)).sum())}")
    bagian.append("")
    bagian.append("Jumlah kosong pada kolom kunci:")
    for nama in KOLOM_KUNCI:
        if nama not in df.columns:
            continue
        jumlah = int(df[nama].isna().sum())
        if nama in ("text", "textTranslated"):
            jumlah_teks = int(teks_kosong(df[nama]).sum())
            bagian.append(
                f"  {nama}: NaN = {jumlah}, kosong termasuk spasi = {jumlah_teks}"
            )
        else:
            bagian.append(f"  {nama}: {jumlah}")

    bagian.append("")
    bagian.append("Sepuluh kolom dengan NaN terbanyak:")
    teratas = kosong.sort_values(ascending=False).head(10)
    for nama, jumlah in teratas.items():
        bagian.append(f"  {nama}: {int(jumlah)}")

    # Pemeriksaan teks ulasan.
    if "text" not in df.columns:
        raise KeyError("Kolom 'text' tidak ditemukan. Audit dihentikan.")

    punya_teks = ~teks_kosong(df["text"])
    teks_bersih = df.loc[punya_teks, "text"].astype(str).str.strip()

    bagian.append(blok("5. Ulasan yang memiliki teks"))
    bagian.append(f"Baris dengan teks pada kolom text: {int(punya_teks.sum())}")
    bagian.append(f"Baris tanpa teks pada kolom text: {int((~punya_teks).sum())}")
    if "textTranslated" in df.columns:
        punya_terjemahan = ~teks_kosong(df["textTranslated"])
        bagian.append(f"Baris dengan isi textTranslated: {int(punya_terjemahan.sum())}")
        bagian.append(
            "Baris tanpa teks asli tetapi punya textTranslated: "
            f"{int((~punya_teks & punya_terjemahan).sum())}"
        )
        bagian.append(
            "textTranslated tidak dipakai sebagai pengganti teks asli pada tahap ini."
        )

    panjang = teks_bersih.str.len()
    bagian.append("")
    bagian.append("Panjang teks asli (hanya baris yang punya teks):")
    if len(panjang) == 0:
        bagian.append("  Tidak ada teks untuk diukur.")
    else:
        bagian.append(f"  Minimum: {int(panjang.min())} karakter")
        bagian.append(f"  Median: {float(panjang.median()):.0f} karakter")
        bagian.append(f"  Maksimum: {int(panjang.max())} karakter")
        bagian.append(f"  Teks sangat pendek (1–3 karakter): {int((panjang <= 3).sum())}")

    bagian.append(blok("6. Duplikat"))
    bagian.append("Pengenal reviewer tidak disimpan dalam dataset.")

    duplikat_teks = int(teks_bersih.duplicated().sum())
    bagian.append(f"Teks ulasan yang muncul lebih dari sekali (salinan tambahan): {duplikat_teks}")
    bagian.append(f"Teks ulasan unik di antara baris yang punya teks: {teks_bersih.nunique()}")
    bagian.append(blok("7. Distribusi rating (kolom stars)"))
    if "stars" not in df.columns:
        bagian.append("Kolom stars tidak ada.")
    else:
        bagian.append(format_hitungan(df["stars"]))
        bagian.append("")
        bagian.append("Distribusi stars hanya pada baris yang punya teks:")
        bagian.append(format_hitungan(df.loc[punya_teks, "stars"]))

    if "rating" in df.columns:
        bagian.append("")
        bagian.append("Distribusi kolom rating (kolom tempat, bukan rating ulasan per baris):")
        bagian.append(format_hitungan(df["rating"]))
        if "stars" in df.columns:
            sama = int((df["rating"] == df["stars"]).sum())
            bagian.append(f"Baris yang nilai rating-nya sama dengan stars: {sama} dari {len(df)}")

    bagian.append(blok("8. Distribusi bahasa"))
    if "originalLanguage" in df.columns:
        bagian.append("originalLanguage (seluruh baris):")
        bagian.append(format_hitungan(df["originalLanguage"]))
        bagian.append("")
        bagian.append("originalLanguage pada baris yang punya teks:")
        bagian.append(format_hitungan(df.loc[punya_teks, "originalLanguage"]))
    else:
        bagian.append("Kolom originalLanguage tidak ada.")

    if "language" in df.columns:
        bagian.append("")
        bagian.append("Kolom language (seluruh baris) — ini berbeda dari originalLanguage:")
        bagian.append(format_hitungan(df["language"]))

    bagian.append(blok("9. Data yang memenuhi kriteria penelitian"))
    bagian.append(
        "Kriteria yang dihitung di sini:\n"
        "  a. originalLanguage bernilai id\n"
        "  b. kolom text terisi (bukan kosong atau hanya spasi)\n"
        "  c. teks unik (kemunculan kedua dan seterusnya tidak dihitung)\n"
        "  d. stars bernilai 1, 2, 4, atau 5 (rating 3 dikeluarkan dari klasifikasi biner)"
    )

    if "originalLanguage" not in df.columns or "stars" not in df.columns:
        bagian.append("Kriteria tidak dapat dihitung karena kolom yang diperlukan tidak lengkap.")
    else:
        bahasa_id = df["originalLanguage"].astype(str).str.strip().str.lower().eq("id")
        kandidat = bahasa_id & punya_teks
        bagian.append(f"Bahasa Indonesia dan punya teks: {int(kandidat.sum())}")

        df_kandidat = df.loc[kandidat].copy()
        df_kandidat["_teks"] = df_kandidat["text"].astype(str).str.strip()
        sebelum_unik = len(df_kandidat)
        df_unik = df_kandidat.drop_duplicates(subset=["_teks"], keep="first")
        bagian.append(
            f"Setelah teks duplikat diabaikan (menyisakan kemunculan pertama): {len(df_unik)}"
        )
        bagian.append(f"Duplikat teks di dalam kandidat bahasa Indonesia: {sebelum_unik - len(df_unik)}")

        stars_numerik = pd.to_numeric(df_unik["stars"], errors="coerce")
        bagian.append("")
        bagian.append("Distribusi stars pada kandidat unik berbahasa Indonesia:")
        bagian.append(format_hitungan(stars_numerik))

        biner = stars_numerik.isin([1, 2, 4, 5])
        netral = stars_numerik.eq(3)
        bukan_1_sampai_5 = ~stars_numerik.isin([1, 2, 3, 4, 5])
        bagian.append("")
        bagian.append(f"Layak klasifikasi biner (bintang 1, 2, 4, 5): {int(biner.sum())}")
        bagian.append(f"Rating 3 (netral, dikeluarkan dari klasifikasi biner): {int(netral.sum())}")
        bagian.append(f"Stars di luar 1–5 atau kosong: {int(bukan_1_sampai_5.sum())}")
        positif = int(stars_numerik.isin([4, 5]).sum())
        negatif = int(stars_numerik.isin([1, 2]).sum())
        bagian.append(f"  di antaranya positif (4 dan 5): {positif}")
        bagian.append(f"  di antaranya negatif (1 dan 2): {negatif}")

    if "publishedAtDate" in df.columns:
        tanggal = pd.to_datetime(df["publishedAtDate"], errors="coerce")
        bagian.append(blok("10. Rentang tanggal ulasan"))
        bagian.append(f"Tanggal yang berhasil dibaca: {int(tanggal.notna().sum())}")
        bagian.append(f"Tanggal yang gagal dibaca atau kosong: {int(tanggal.isna().sum())}")
        if tanggal.notna().any():
            bagian.append(f"Tanggal paling awal: {tanggal.min()}")
            bagian.append(f"Tanggal paling akhir: {tanggal.max()}")

    laporan = "\n".join(bagian) + "\n"
    PATH_LAPORAN.parent.mkdir(parents=True, exist_ok=True)
    PATH_LAPORAN.write_text(laporan, encoding="utf-8")

    print(laporan)
    print(f"Laporan disimpan di: {PATH_LAPORAN.name}")


if __name__ == "__main__":
    main()
