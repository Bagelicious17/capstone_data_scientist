# 📄 CV Classification (NLP) - Capstone Project

Proyek ini adalah pipeline pemrosesan teks berbasis *Natural Language Processing* (NLP) untuk melakukan klasifikasi dokumen Curriculum Vitae (CV) secara otomatis berdasarkan keahlian dan kategori industri.

## 📁 Struktur Direktori

```text
capstone_data_scientist/
├── data/
│   ├── raw/                 # Data mentah yang tidak boleh diubah (Dataset_CV.xlsx)
│   └── processed/           # Data hasil cleaning yang Siap Model (Dataset_NLP_Siap_Model.xlsx)
├── reports/                 # Folder untuk menyimpan hasil evaluasi, plot, atau metrik model
├── src/                     # Source code operasional utama
│   ├── data_cleaning.py     # Skrip pipeline pembersihan data & NLP preprocessing (NLTK)
│   └── data_wrangling_EDA.ipynb # Notebook untuk Exploratory Data Analysis (EDA)
├── testing/                 # Folder untuk eksperimen script / test cases (test1.py)
├── requirements.txt         # Daftar dependencies library Python
└── README.md                # Dokumentasi proyek
```

## 🚀 Cara Menjalankan Pipeline

### 1. Instalasi Dependencies
Pastikan Anda sudah menginstall Python. Buka terminal/Command Prompt dan jalankan perintah berikut untuk menginstall semua library yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Data Cleaning & NLP Preprocessing
Skrip `data_cleaning.py` dirancang untuk membersihkan data mentah secara modular.
Tahapan NLP yang diaplikasikan meliputi:
- **Lowercasing**: Menyeragamkan seluruh teks.
- **Penghapusan PII / Noise**: Membuang alamat email, URL, dan nomor/simbol yang tidak relevan.
- **Penghapusan Stopwords**: Membuang kata hubung bahasa Inggris menggunakan kamus `nltk`.
- **Lemmatization**: Mengembalikan kata-kata ke bentuk dasarnya (*base form*) menggunakan `WordNetLemmatizer`.

Jalankan perintah ini di dalam direktori `capstone_data_scientist`:
```bash
python src/data_cleaning.py
```
*(Hasil olahan akan otomatis tersimpan di folder `data/processed/` dengan format Excel yang rapi).*

### 3. Eksplorasi Data (EDA)
Untuk melihat visualisasi analisis distribusi kategori profesi, persebaran kata dominan, dan insight lainnya, Anda dapat menjalankan notebook EDA:
```bash
jupyter notebook src/data_wrangling_EDA.ipynb
```

## ⚙️ Tahap Selanjutnya (Future Work)
- **Modeling**: Melakukan ekstraksi fitur numerik (misalnya menggunakan `TF-IDF Vectorizer` atau *Word Embeddings*) dan melatih algoritma Machine Learning (seperti Naive Bayes, SVM, atau Random Forest) untuk memprediksi kategori.
- **Evaluation**: Mengukur kualitas model melalui parameter metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *Confusion Matrix*.

---
*Dibuat untuk kebutuhan Capstone Project Coding Camp DBS Foundation.*
