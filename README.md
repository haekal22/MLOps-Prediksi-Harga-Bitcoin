# MLOps-Prediksi-Harga-Bitcoin
# Sistem Prediksi Harga Bitcoin Berbasis MLOps

## Tujuan Proyek

Proyek ini bertujuan untuk membangun sistem prediksi harga Bitcoin berbasis Time Series
dengan pendekatan MLOps. Sistem dirancang untuk mendukung continuous training
guna menghadapi data drift pada pasar kripto yang dinamis.

---

## Struktur Direktori

Berikut struktur proyek yang digunakan:

```
MLOps-Prediksi-Harga-Bitcoin/
│
├── data/           → Berkaitan dengan data yang diambil (data ada versioningnya)
│   ├── raw/        → Data mentah dari API
│   └── processed/  → Data yang sudah dibersihkan
│
├── models/         → Penyimpanan model hasil training
├── notebooks/      → Eksperimen dan Exploratory Data Analysis (EDA)
├── src/            → Source code utama (data ingestion, training, dll)
├── config/         → File konfigurasi (parameter, setting, dsb)
├── tests/          → Unit testing
├── docs/           → Dokumentasi tambahan
│
├── requirements.txt
└── README.md
```
---

## Cara Menjalankan Project di Codespaces

1. Buka repository di GitHub
2. Klik tombol **Code**
3. Pilih tab **Codespaces**
4. Klik **Create Codespace**
5. Tunggu environment selesai loading
6. Install dependency dengan:

```
pip install -r requirements.txt
```

7. Jalankan script Python sesuai kebutuhan, misalnya:

```
python src/hello.py
```

---

## Workflow Git

Project ini menggunakan GitHub Flow:

- Buat branch fitur dari `main`
- Lakukan commit di branch fitur
- Buat Pull Request
- Merge ke `main` setelah divalidasi
