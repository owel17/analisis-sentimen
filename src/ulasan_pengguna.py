"""
Penyimpanan ulasan pengunjung untuk tiap kolam.

Berkas CSV ini terpisah dari dataset penelitian. Skrip pelatihan, evaluasi,
dan halaman statistik penelitian tidak membaca isi berkas ini.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from konfigurasi import PATH_ULASAN_PENGGUNA

DAFTAR_KOLAM = [
    "Kolam Mayangsari",
    "Kolam Wangsadipa",
    "Kolam Pulosari",
    "Kolam Leuwisari",
    "Sariater Waterpark",
]

KOLOM_WAKTU = "waktu"
KOLOM_NAMA = "nama_tampilan"
KOLOM_KOLAM = "kolam"
KOLOM_BINTANG = "bintang"
KOLOM_TEKS_ULASAN = "teks"
KOLOM_LABEL_ULASAN = "label_sentimen"
KOLOM_URUTAN = [
    KOLOM_WAKTU,
    KOLOM_NAMA,
    KOLOM_KOLAM,
    KOLOM_BINTANG,
    KOLOM_TEKS_ULASAN,
    KOLOM_LABEL_ULASAN,
]
LABEL_DITERIMA = frozenset({"positif", "negatif"})
PANJANG_NAMA = 40
ZONA_WAKTU = ZoneInfo("Asia/Jakarta")


def muat_ulasan() -> pd.DataFrame:
    """Baca ulasan pengunjung. Kembalikan tabel kosong jika berkas belum ada."""
    if not PATH_ULASAN_PENGGUNA.exists():
        return pd.DataFrame(columns=KOLOM_URUTAN)

    data = pd.read_csv(PATH_ULASAN_PENGGUNA, encoding="utf-8-sig")
    for kolom in KOLOM_URUTAN:
        if kolom not in data.columns:
            data[kolom] = pd.NA
    data = data[KOLOM_URUTAN].copy()
    data[KOLOM_NAMA] = data[KOLOM_NAMA].fillna("").astype(str)
    data[KOLOM_BINTANG] = pd.to_numeric(data[KOLOM_BINTANG], errors="coerce")
    return data


def tambah_ulasan(kolam: str, bintang: int, teks: str, label: str, nama: str) -> None:
    """Tambahkan satu ulasan. Nama kosong disimpan kosong dan ditampilkan lain waktu."""
    if kolam not in DAFTAR_KOLAM:
        raise ValueError(f"Kolam tidak dikenal: {kolam}")
    if int(bintang) not in range(1, 6):
        raise ValueError("Bintang harus 1 sampai 5.")
    if label not in LABEL_DITERIMA:
        raise ValueError(f"Label sentimen tidak dikenal: {label}")
    teks = "" if teks is None else str(teks).strip()
    if not teks:
        raise ValueError("Teks ulasan kosong.")
    nama = "" if nama is None else " ".join(str(nama).split())
    if len(nama) > PANJANG_NAMA:
        raise ValueError(f"Nama tampilan paling panjang {PANJANG_NAMA} karakter.")

    baris = pd.DataFrame(
        [
            {
                KOLOM_WAKTU: datetime.now(ZONA_WAKTU).strftime("%Y-%m-%d %H:%M"),
                KOLOM_NAMA: nama,
                KOLOM_KOLAM: kolam,
                KOLOM_BINTANG: int(bintang),
                KOLOM_TEKS_ULASAN: teks,
                KOLOM_LABEL_ULASAN: label,
            }
        ]
    )
    berkas_baru = not PATH_ULASAN_PENGGUNA.exists() or PATH_ULASAN_PENGGUNA.stat().st_size == 0
    PATH_ULASAN_PENGGUNA.parent.mkdir(parents=True, exist_ok=True)
    baris.to_csv(
        PATH_ULASAN_PENGGUNA,
        mode="a",
        header=berkas_baru,
        index=False,
        encoding="utf-8-sig",
    )


def nama_tampilan(nama: str) -> str:
    """Nama yang diketik pengunjung, atau Pengunjung bila dikosongkan."""
    bersih = "" if nama is None else str(nama).strip()
    return bersih or "Pengunjung"
