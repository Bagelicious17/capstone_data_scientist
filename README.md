# 📄 CV Classification (NLP) - Capstone Project

Proyek ini adalah pipeline pemrosesan teks berbasis *Natural Language Processing* (NLP) untuk melakukan klasifikasi dokumen Curriculum Vitae (CV) secara otomatis berdasarkan keahlian dan kategori industri.

**🌐 Live Dashboard:** [https://capstonedatascientist-yzdfoknpqbiznrkgvuz6p8.streamlit.app/](https://capstonedatascientist-yzdfoknpqbiznrkgvuz6p8.streamlit.app/)

## 📊 Informasi Dataset
Dataset mentah (`Dataset_CV.xlsx`) terdiri dari sekumpulan teks *resume* profesional yang mencakup 8 kategori jabatan utama (seperti *Data Science*, *Software Engineer*, *Business Analyst*, dll). Dataset ini telah melalui tahap anonimisasi dan disiapkan khusus untuk melatih model agar dapat mengenali pola kosakata dari masing-masing profesi.

## 📁 Struktur Direktori

```text
capstone_data_scientist/
├── data/
│   ├── raw/                 # Data mentah yang tidak boleh diubah (Dataset_CV.xlsx)
│   └── processed/           # Data hasil cleaning yang Siap Model (Dataset_NLP_Siap_Model.xlsx)
├── reports/                 # Folder untuk menyimpan hasil evaluasi, plot, atau metrik model
├── src/                     # Source code operasional utama
│   ├── data_cleaning.py     # Skrip pipeline pembersihan data & NLP preprocessing (NLTK)
│   ├── dashboard.py         # Dashboard interaktif (Streamlit) untuk profil dataset
│   └── data_wrangling_EDA.ipynb # Notebook untuk Exploratory Data Analysis (EDA)
├── testing/                 # Folder untuk eksperimen script / test cases (test1.py)
├── requirements.txt         # Daftar dependencies library Python
└── README.md                # Dokumentasi proyek
```

## 🚀 Cara Menjalankan Pipeline

### 1. Instalasi Dependencies
**Prasyarat:** Pastikan Anda menggunakan **Python 3.9 atau lebih baru**.

Sangat disarankan untuk membuat *Virtual Environment* terlebih dahulu agar *library* proyek ini terisolasi dengan rapi. Buka terminal/Command Prompt dan jalankan urutan perintah berikut:

```bash
# 1. Membuat virtual environment (bernama 'venv')
python -m venv venv

# 2. Mengaktifkan virtual environment
# Untuk pengguna Windows:
venv\Scripts\activate
# Untuk pengguna Mac/Linux:
# source venv/bin/activate

# 3. Menginstall seluruh library yang dibutuhkan
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

### 4. Menjalankan Dashboard Analisis Data
Untuk memberikan visualisasi interaktif mengenai kualitas dataset (distribusi kelas, outlier, dan *word cloud*) kepada tim AI Engineer, jalankan *dashboard* berbasis Streamlit:
```bash
streamlit run src/dashboard.py
```
*(Dashboard akan otomatis terbuka di browser melalui `http://localhost:8501`)*

**Catatan:** Anda juga dapat langsung melihat versi *live* yang sudah di-*deploy* di sini: 
👉 [https://capstonedatascientist-yzdfoknpqbiznrkgvuz6p8.streamlit.app/](https://capstonedatascientist-yzdfoknpqbiznrkgvuz6p8.streamlit.app/)

## ⚙️ Tahap Selanjutnya (Future Work)
- **Modeling**: Melakukan ekstraksi fitur numerik (misalnya menggunakan `TF-IDF Vectorizer` atau *Word Embeddings*) dan melatih algoritma Machine Learning (seperti Naive Bayes, SVM, atau Random Forest) untuk memprediksi kategori.
- **Evaluation**: Mengukur kualitas model melalui parameter metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan *Confusion Matrix*.

---
*Dibuat untuk kebutuhan Capstone Project Coding Camp DBS Foundation.*
