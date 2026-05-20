"""
==============================================================
 data_cleaning.py — Capstone Project: CV Classification (NLP)
==============================================================
 Pipeline terbaru:
   TAHAP 1 : Load dataset mentah dari data/raw/Dataset_CV.xlsx
   TAHAP 2 : Standardisasi kategori jabatan modern
   TAHAP 3 : Pembersihan & normalisasi teks
   TAHAP 4 : Deduplikasi dan balancing sederhana
   TAHAP 5 : Feature engineering
   TAHAP 6 : Export dataset siap dashboard/model

 Kategori tambahan yang dimasukkan:
   - AI Engineer
   - ML Engineer
   - Front End Developer
   - Back End Developer
   - Full-Stack Developer
   - Mobile Developer
   - Project Management

 Cara pakai:
   python src/data_cleaning.py
   Output utama:
   data/processed/Dataset_NLP_Siap_Model_V3.xlsx
==============================================================
"""

import os
import re
import unicodedata
from typing import Iterable, Optional

import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:  # pragma: no cover - fallback untuk environment tanpa NLTK
    nltk = None
    stopwords = None
    WordNetLemmatizer = None


# ==============================================================
# KONFIGURASI PATH
# ==============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'Dataset_CV.xlsx')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'Dataset_NLP_Siap_Model_V3.xlsx')
OUTPUT_COMPAT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'Dataset_NLP_Siap_Model_V2.xlsx')
SAMPLE_FILE = os.path.join(BASE_DIR, 'sample.csv')

TARGET_SAMPLE_PER_CATEGORY = 320
RANDOM_STATE = 42

CRITICAL_CATEGORIES = [
    'AI Engineer',
    'ML Engineer',
    'Front End Developer',
    'Back End Developer',
    'Full-Stack Developer',
    'Mobile Developer',
    'Project Management',
]

ROLE_GROUP = {
    'AI Engineer': 'Data & AI',
    'ML Engineer': 'Data & AI',
    'Business Analyst': 'Business & Product',
    'Front End Developer': 'Software Engineering',
    'Back End Developer': 'Software Engineering',
    'Full-Stack Developer': 'Software Engineering',
    'Mobile Developer': 'Software Engineering',
    'Network Security Engineer': 'Cybersecurity & Infrastructure',
    'Project Management': 'Project & Management',
}

# Mapping target kategori dari kategori sumber + keyword.
FRONTEND_SOURCES = {'React Developer', 'Web Designing'}
BACKEND_SOURCES = {'Java Developer', 'SQL Developer', 'SAP Developer', 'DotNet Developer', 'Database', 'ETL Developer', 'Python Developer'}
AI_ML_SOURCES = {'Data Science', 'Python Developer'}
MOBILE_SOURCES = {'React Developer', 'Java Developer', 'DotNet Developer', 'Blockchain', 'Information Technology', 'Testing', 'Web Designing'}
FULLSTACK_SOURCES = {'Java Developer', 'DotNet Developer', 'React Developer', 'Web Designing', 'Python Developer', 'Blockchain', 'Information Technology'}
PM_SOURCES = {'PMO', 'Management', 'Operations Manager', 'Information Technology', 'Consultant'}

FRONTEND_TERMS = [
    'react', 'redux', 'javascript', 'typescript', 'html', 'css', 'front end',
    'frontend', 'ui', 'ux', 'angular', 'vue', 'responsive', 'web interface'
]
BACKEND_TERMS = [
    'backend', 'back end', 'java', 'spring', 'node', 'nodejs', 'api', 'rest',
    'database', 'sql', 'server', 'asp.net', '.net', 'c#', 'django', 'flask',
    'microservice', 'etl', 'oracle', 'sap hana'
]
MOBILE_TERMS = [
    'react native', 'android', 'ios', 'flutter', 'kotlin', 'xcode',
    'objective-c', 'objective c', 'mobile app', 'mobile apps',
    'mobile application', 'mobile developer'
]
AI_TERMS = [
    'artificial intelligence', 'deep learning', 'neural', 'computer vision',
    'natural language', 'nlp', 'generative ai', 'ai model', 'tensorflow',
    'pytorch', 'keras'
]
ML_TERMS = [
    'machine learning', 'predictive', 'classification', 'regression',
    'clustering', 'algorithm', 'modeling', 'model', 'scikit', 'sklearn',
    'feature engineering', 'random forest', 'svm', 'naive bayes'
]
PM_TERMS = [
    'pmo', 'project manager', 'project management', 'program manager',
    'scrum master', 'agile', 'jira', 'stakeholder', 'budget', 'schedule',
    'timeline', 'risk management', 'resource allocation'
]

FALLBACK_STOPWORDS = {
    'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'with', 'as', 'at',
    'by', 'an', 'be', 'this', 'that', 'are', 'was', 'were', 'have', 'has',
    'from', 'or', 'it', 'not', 'but', 'we', 'you', 'they', 'he', 'she', 'i',
    'me', 'my', 'your', 'our', 'their', 'its', 'also', 'will', 'been', 'which',
    'who', 'more', 'into', 'can', 'all', 'one', 'about', 'so', 'if', 'up',
    'out', 'do', 'his', 'her', 'had', 'may', 'cv', 'resume', 'work',
    'experience', 'years', 'year', 'skills', 'working', 'used', 'using',
    'team', 'project', 'projects', 'company', 'worked', 'able', 'well', 'new',
    'good', 'strong', 'etc', 'per', 'including', 'within', 'across', 'through',
    'over', 'present', 'summary', 'education', 'email', 'phone', 'address',
    'linkedin', 'objective', 'professional', 'skill'
}


def load_stopwords() -> set:
    """Gunakan NLTK jika tersedia; jika tidak, pakai fallback lokal agar script tetap jalan offline."""
    if nltk is None or stopwords is None:
        return FALLBACK_STOPWORDS

    try:
        nltk.data.find('corpora/stopwords')
        return set(stopwords.words('english')) | FALLBACK_STOPWORDS
    except Exception:
        # Offline-safe fallback: do not download resources during grading/deployment.
        return FALLBACK_STOPWORDS


def load_lemmatizer():
    if nltk is None or WordNetLemmatizer is None:
        return None
    try:
        nltk.data.find('corpora/wordnet')
        return WordNetLemmatizer()
    except Exception:
        # Offline-safe fallback: skip lemmatization if WordNet is unavailable.
        return None


STOPWORDS = load_stopwords()
LEMMATIZER = load_lemmatizer()


# ==============================================================
# HELPER
# ==============================================================

def contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\ufeff', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def bersihkan_teks(teks_mentah: object) -> str:
    """
    Bersihkan teks untuk model NLP:
    - Lowercase
    - Hapus email, URL, angka, dan simbol
    - Hapus stopwords
    - Lemmatization jika resource tersedia
    """
    teks = normalize_text(teks_mentah).lower()
    teks = re.sub(r'\S+@\S+', ' ', teks)
    teks = re.sub(r'http\S+|www\.\S+', ' ', teks)
    teks = re.sub(r'[^a-z\s]', ' ', teks)
    words = teks.split()

    cleaned_words = []
    for word in words:
        if word in STOPWORDS or len(word) < 3:
            continue
        if LEMMATIZER is not None:
            try:
                word = LEMMATIZER.lemmatize(word)
            except Exception:
                pass
        cleaned_words.append(word)

    return ' '.join(cleaned_words)


def assign_updated_category(original_category: str, text: object) -> Optional[str]:
    """
    Kategori diperbarui agar dataset lebih relevan dengan role teknologi saat ini.
    Urutan rule penting: role spesifik (mobile/full-stack/AI/ML) diprioritaskan
    sebelum fallback kategori yang lebih umum.
    """
    cat = str(original_category).strip()
    text_l = normalize_text(text).lower()

    # Kategori awal yang tetap dipertahankan.
    if cat == 'Business Analyst':
        return 'Business Analyst'
    if cat == 'Network Security Engineer':
        return 'Network Security Engineer'

    # Kategori tambahan krusial.
    if cat == 'PMO' or (cat in PM_SOURCES and contains_any(text_l, PM_TERMS)):
        return 'Project Management'

    if cat in MOBILE_SOURCES and contains_any(text_l, MOBILE_TERMS):
        return 'Mobile Developer'

    if cat in FULLSTACK_SOURCES and (
        'full stack' in text_l or 'full-stack' in text_l or
        (cat not in FRONTEND_SOURCES and contains_any(text_l, FRONTEND_TERMS) and contains_any(text_l, BACKEND_TERMS))
    ):
        return 'Full-Stack Developer'

    if cat in AI_ML_SOURCES and contains_any(text_l, AI_TERMS):
        return 'AI Engineer'

    if cat in AI_ML_SOURCES and contains_any(text_l, ML_TERMS):
        return 'ML Engineer'

    if cat in FRONTEND_SOURCES:
        return 'Front End Developer'

    if cat in BACKEND_SOURCES:
        return 'Back End Developer'

    return None


# ==============================================================
# PIPELINE
# ==============================================================

def load_data() -> pd.DataFrame:
    print('=' * 72)
    print(' CAPSTONE — CV CLASSIFICATION DATASET ENRICHMENT PIPELINE')
    print('=' * 72)

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f'File raw tidak ditemukan: {RAW_FILE}')

    df = pd.read_excel(RAW_FILE, engine='openpyxl')
    required = {'Category', 'Text'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Kolom wajib tidak ditemukan: {missing}')

    df = df[['Category', 'Text']].copy()
    df['Original_Category'] = df['Category'].astype(str).str.strip()
    df['Text'] = df['Text'].apply(normalize_text)
    df = df[df['Text'].str.len() > 0].copy()

    print(f'RAW FILE       : {RAW_FILE}')
    print(f'Jumlah baris   : {len(df):,}')
    print(f'Kategori raw   : {df["Original_Category"].nunique()}')
    return df


def standardisasi_kategori(df: pd.DataFrame) -> pd.DataFrame:
    print('\nTAHAP 2 — STANDARDISASI KATEGORI')
    print('-' * 44)
    df = df.copy()
    df['Category'] = df.apply(lambda row: assign_updated_category(row['Original_Category'], row['Text']), axis=1)
    df = df[df['Category'].notna()].copy()
    df['Role_Group'] = df['Category'].map(ROLE_GROUP).fillna('Other')

    print('Kategori hasil mapping:')
    print(df['Category'].value_counts().to_string())
    return df


def terapkan_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    print('\nTAHAP 3 — PEMBERSIHAN TEKS')
    print('-' * 44)
    df = df.copy()
    df['Clean_Text'] = df['Text'].apply(bersihkan_teks)
    df = df[df['Clean_Text'].str.split().str.len() >= 20].copy()
    print(f'Baris setelah cleaning valid: {len(df):,}')
    return df


def deduplikasi_dan_balancing(df: pd.DataFrame) -> pd.DataFrame:
    print('\nTAHAP 4 — DEDUPLIKASI & BALANCING')
    print('-' * 44)
    before = len(df)
    df = df.drop_duplicates(subset=['Text']).copy()
    print(f'Duplikat dihapus: {before - len(df):,}')

    parts = []
    for category, group in df.groupby('Category', sort=True):
        if len(group) > TARGET_SAMPLE_PER_CATEGORY:
            group = group.sample(TARGET_SAMPLE_PER_CATEGORY, random_state=RANDOM_STATE)
        parts.append(group)

    df_balanced = pd.concat(parts, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print('Distribusi akhir:')
    print(df_balanced['Category'].value_counts().to_string())
    return df_balanced


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print('\nTAHAP 5 — FEATURE ENGINEERING')
    print('-' * 44)
    df = df.copy()
    df.insert(0, 'ID_CV', ['CV_' + str(i).zfill(4) for i in range(1, len(df) + 1)])
    df['Word_Count'] = df['Clean_Text'].apply(lambda x: len(str(x).split()))
    df['Char_Count'] = df['Clean_Text'].apply(lambda x: len(str(x)))

    print(f'Total CV      : {len(df):,}')
    print(f'Total kategori: {df["Category"].nunique()}')
    print(f'Rata-rata kata: {df["Word_Count"].mean():.0f}')
    return df


def export_dataset(df: pd.DataFrame) -> pd.DataFrame:
    print('\nTAHAP 6 — EXPORT DATASET')
    print('-' * 44)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    final_columns = [
        'ID_CV', 'Category', 'Original_Category', 'Role_Group',
        'Word_Count', 'Char_Count', 'Clean_Text', 'Text'
    ]
    df_final = df[final_columns].copy().sort_values(['Category', 'ID_CV']).reset_index(drop=True)
    df_final['ID_CV'] = ['CV_' + str(i).zfill(4) for i in range(1, len(df_final) + 1)]

    df_final.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
    df_final.to_excel(OUTPUT_COMPAT_FILE, index=False, engine='openpyxl')
    df_final.head(200).to_csv(SAMPLE_FILE, index=False, encoding='utf-8')

    print(f'Output utama       : {OUTPUT_FILE}')
    print(f'Output kompatibel  : {OUTPUT_COMPAT_FILE}')
    print(f'Sample CSV         : {SAMPLE_FILE}')
    return df_final


def main():
    df = load_data()
    df = standardisasi_kategori(df)
    df = terapkan_cleaning(df)
    df = deduplikasi_dan_balancing(df)
    df_final = feature_engineering(df)
    df_final = export_dataset(df_final)

    print('\n' + '=' * 72)
    print(' PIPELINE SELESAI — RINGKASAN HASIL')
    print('=' * 72)
    print(f'Total CV final : {len(df_final):,}')
    print(f'Kategori final : {df_final["Category"].nunique()}')
    print('\nDistribusi kategori final:')
    print(df_final['Category'].value_counts().to_string())
    print('=' * 72)


if __name__ == '__main__':
    main()
