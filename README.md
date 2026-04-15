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

---

## Data Ingestion

Proses pengambilan data dilakukan menggunakan API dari CoinGecko melalui script:

```
python src/ingest_data.py
```

Fitur:
- Mengambil data Bitcoin:
- Harga (price)
- Market Cap
- Volume
- Data dikonversi ke waktu WIB
- Data disimpan dalam format CSV
- Menggunakan timestamp pada nama file untuk menghindari overwrite
- Mendukung pengambilan data secara berkala (simulasi continual learning)
Output:
Lokasi: data/raw/
Contoh file:
btc_market_20260404_120000.csv

---

## Data Preprocessing

Proses preprocessing dilakukan menggunakan script:

```
python src/preprocess.py
```

Proses yang dilakukan:
- Membaca file terbaru dari folder data/raw/
- Konversi kolom datetime ke format yang sesuai
- Mengurutkan data berdasarkan waktu
- Menghapus data duplikat
- Menghapus missing values
- Reset index
Output:
Lokasi: data/processed/
File hasil:
btc_clean.csv

---

## Cara Menjalankan Pipeline

Jalankan secara berurutan:

```
python src/ingest_data.py 
python src/preprocess.py
```

## Data Versioning dengan DVC

Proyek ini menggunakan DVC (Data Version Control) untuk melakukan versioning dataset tanpa menyimpan file besar ke Git.

### Inisialisasi DVC
```
dvc init
```

### Tracking dataset
Dataset hasil preprocessing dilacak menggunakan:

```
dvc add data/processed/btc_features.csv
```


Hasil:
- File `.dvc` dibuat
- Git hanya menyimpan metadata dataset

---

## Simulasi Continual Learning

Dataset diperbarui melalui proses ingestion:

```
python src/ingest_data.py
python src/preprocess.py
```

Setelah data berubah, dataset dilacak kembali:

```
dvc add data/processed/btc_features.csv
```


---

## Melihat Perubahan Dataset

Cek status dataset:
```
dvc status
```

Cek perbedaan versi:
```
dvc diff
```

---

## Versioning dengan Git + DVC

Setiap perubahan dataset dicatat dengan Git:
```
git add data/processed/btc_features.csv.dvc
git commit -m "update dataset version"
```

---

## Tujuan Penggunaan DVC

- Menghindari penyimpanan dataset besar di Git
- Melakukan versioning dataset (v1, v2, dst)
- Mendukung continuous learning
- Memudahkan tracking perubahan data ML

---

## Alur Versioning DVC

Dataset awal → dvc add → commit (v1)  
↓  
Data baru masuk  
↓  
dvc add lagi → commit (v2)  
↓  
dvc diff untuk melihat perubahan
