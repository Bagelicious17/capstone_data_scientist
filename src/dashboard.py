"""
==============================================================
 dashboard.py — Capstone Project: CV Classification (NLP)
==============================================================
 Cara pakai:
   pip install streamlit plotly wordcloud matplotlib openpyxl
   streamlit run src/dashboard.py
==============================================================
"""

import os
import re
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud


# ==============================================================
# KONFIGURASI HALAMAN
# ==============================================================

st.set_page_config(
    page_title="CV Classification Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

if st.session_state.theme == 'Light':
    bg_main = '#ffffff'
    bg_sidebar = '#f9fafb'
    bg_card = '#ffffff'
    border_color = '#e5e7eb'
    text_main = '#1e3a5f'
    text_sub = '#6b7280'
    text_muted = '#9ca3af'
    plotly_font = '#374151'
    plotly_grid = '#f3f4f6'
    wordcloud_bg = 'white'
else:
    bg_main = '#0e1117'
    bg_sidebar = '#262730'
    bg_card = '#1f2937'
    border_color = '#374151'
    text_main = '#f9fafb'
    text_sub = '#9ca3af'
    text_muted = '#6b7280'
    plotly_font = '#e5e7eb'
    plotly_grid = '#374151'
    wordcloud_bg = '#0e1117'

st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_main}; }}
    [data-testid="stHeader"] {{ background-color: transparent; }}
    html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    .card {{
        background-color: {bg_card};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .card-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {text_main};
        margin: 0;
        line-height: 1.2;
    }}
    .card-label {{
        font-size: 0.78rem;
        color: {text_sub};
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .card-sub {{
        font-size: 0.73rem;
        color: {text_muted};
        margin: 2px 0 0 0;
    }}
    .section-header {{
        font-size: 0.88rem;
        font-weight: 600;
        color: {text_main};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border-bottom: 2px solid {text_main};
        padding-bottom: 6px;
        margin: 24px 0 14px 0;
    }}
    .page-title {{
        font-size: 1.55rem;
        font-weight: 700;
        color: {text_main};
        margin: 0;
    }}
    .page-subtitle {{
        font-size: 0.86rem;
        color: {text_sub};
        margin: 4px 0 0 0;
    }}
    .divider {{ height: 1px; background: {border_color}; margin: 14px 0; }}
    .badge {{
        display: inline-block;
        padding: 4px 8px;
        margin: 3px 4px 3px 0;
        border-radius: 999px;
        border: 1px solid {border_color};
        color: {text_sub};
        font-size: 0.75rem;
    }}
    .insight-box {{
        background-color: {bg_card};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 0.84rem;
        color: {text_main};
        min-height: 112px;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        border-right: 1px solid {border_color};
    }}
    .sidebar-label {{
        font-size: 0.73rem;
        font-weight: 600;
        color: {text_sub};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0px; border-bottom: 2px solid {border_color}; }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.83rem;
        font-weight: 500;
        color: {text_sub};
        padding: 10px 20px;
        border-radius: 0;
        background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {text_main} !important;
        border-bottom: 2px solid {text_main} !important;
        font-weight: 600 !important;
    }}
    .stDownloadButton > button {{
        background-color: #1e3a5f;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.83rem;
    }}
    .stDownloadButton > button:hover {{ background-color: #2d5282; }}
    .stApp, [data-testid="stAppViewContainer"] {{ color: {text_main} !important; }}
    [data-testid="stMarkdownContainer"] *, [data-testid="stText"] *, label, p, span, li, h1, h2, h3, h4, h5, h6 {{
        color: {text_main};
    }}
    .card-label, .sidebar-label {{ color: {text_sub} !important; }}
    .card-sub {{ color: {text_muted} !important; }}
    .stTabs [data-baseweb="tab"] {{ color: {text_sub} !important; }}
    .stTabs [aria-selected="true"] {{ color: {text_main} !important; }}
</style>
""", unsafe_allow_html=True)


# ==============================================================
# PALET WARNA & KONFIGURASI
# ==============================================================

COLORS = [
    '#1e3a5f', '#2e7d6e', '#c0392b', '#7d3c98', '#d4860b',
    '#1a6b9a', '#2e6b3e', '#8b5e3c', '#6f42c1', '#0f766e',
    '#b45309', '#be123c', '#0369a1', '#4d7c0f', '#7c3aed'
]

CRITICAL_CATEGORIES = [
    'AI Engineer',
    'ML Engineer',
    'Front End Developer',
    'Back End Developer',
    'Full-Stack Developer',
    'Mobile Developer',
    'Project Management',
]

ROLE_GROUP_MAP = {
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

LAYOUT = dict(
    font=dict(family='Inter, Segoe UI, sans-serif', size=12, color=plotly_font),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=40, b=40, l=20, r=20),
    xaxis=dict(showgrid=True, gridcolor=plotly_grid, zeroline=False, linecolor=border_color, automargin=True),
    yaxis=dict(showgrid=True, gridcolor=plotly_grid, zeroline=False, linecolor=border_color, automargin=True),
)


# ==============================================================
# LOAD DATA
# ==============================================================

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, 'data', 'processed', 'Dataset_NLP_Siap_Model_V3.xlsx'),
        os.path.join(base_dir, 'data', 'processed', 'Dataset_NLP_Siap_Model_V2.xlsx'),
        os.path.join(base_dir, 'data', 'processed', 'Dataset_NLP_Siap_Model.xlsx'),
        'data/processed/Dataset_NLP_Siap_Model_V3.xlsx',
        'data/processed/Dataset_NLP_Siap_Model_V2.xlsx',
        'Dataset_NLP_Siap_Model_V3.xlsx',
    ]

    for path in candidates:
        if os.path.exists(path):
            df_loaded = pd.read_excel(path, engine='openpyxl')
            required_cols = {'ID_CV', 'Category', 'Word_Count', 'Clean_Text', 'Text'}
            missing = required_cols - set(df_loaded.columns)
            if missing:
                raise ValueError(f"Dataset ditemukan tetapi kolom wajib hilang: {missing}")
            df_loaded['Category'] = df_loaded['Category'].astype(str).str.strip()
            df_loaded['Clean_Text'] = df_loaded['Clean_Text'].fillna('').astype(str)
            df_loaded['Text'] = df_loaded['Text'].fillna('').astype(str)
            df_loaded['Word_Count'] = pd.to_numeric(df_loaded['Word_Count'], errors='coerce').fillna(0).astype(int)
            if 'Char_Count' not in df_loaded.columns:
                df_loaded['Char_Count'] = df_loaded['Clean_Text'].apply(lambda x: len(str(x)))
            if 'Original_Category' not in df_loaded.columns:
                df_loaded['Original_Category'] = df_loaded['Category']
            if 'Role_Group' not in df_loaded.columns:
                df_loaded['Role_Group'] = df_loaded['Category'].map(ROLE_GROUP_MAP).fillna('Other')
            return df_loaded, path
    return None, None


@st.cache_data
def get_top_words(texts, n=50):
    stopwords = {
        'the','and','to','of','a','in','for','is','on','with','as','at','by',
        'an','be','this','that','are','was','have','has','from','or','it','not',
        'but','we','you','they','he','she','i','me','my','your','our','their',
        'its','also','will','been','which','who','more','into','can','all','one',
        'about','so','if','up','out','do','his','her','had','may','cv','resume',
        'work','experience','years','year','skills','working','used','using',
        'team','project','projects','company','worked','able','well','new','good',
        'strong','etc','per','including','within','across','through','over',
        'present','summary','education','email','phone','address','linkedin',
        'objective','professional','skill'
    }
    text_joined = ' '.join(pd.Series(texts).dropna().astype(str).str.lower())
    words = re.findall(r'\b[a-z]{3,}\b', text_joined)
    return Counter([w for w in words if w not in stopwords]).most_common(n)


df, data_path = load_data()
if df is None:
    st.error("File dataset tidak ditemukan. Jalankan `python src/data_cleaning.py` atau pastikan file V3/V2 ada di `data/processed/`.")
    st.stop()

cat_order = df['Category'].value_counts().index.tolist()
cat_list = sorted(cat_order)
role_groups = sorted(df['Role_Group'].dropna().unique().tolist())


# ==============================================================
# SIDEBAR
# ==============================================================

with st.sidebar:
    st.markdown('<p class="sidebar-label">Tema Tampilan</p>', unsafe_allow_html=True)
    tema = st.radio("Tema", ["Light", "Dark"], index=0 if st.session_state.theme == 'Light' else 1, horizontal=True, label_visibility="collapsed")
    if tema != st.session_state.theme:
        st.session_state.theme = tema
        st.rerun()

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Filter Role Group</p>', unsafe_allow_html=True)
    group_dipilih = st.multiselect(
        label="Role Group",
        options=role_groups,
        default=role_groups,
        label_visibility="collapsed"
    )

    st.markdown('<p class="sidebar-label" style="margin-top:16px;">Filter Kategori</p>', unsafe_allow_html=True)
    kategori_dipilih = st.multiselect(
        label="Kategori",
        options=cat_list,
        default=cat_list,
        label_visibility="collapsed"
    )

    st.markdown('<p class="sidebar-label" style="margin-top:16px;">Rentang Word Count</p>', unsafe_allow_html=True)
    wc_min, wc_max = int(df['Word_Count'].min()), int(df['Word_Count'].max())
    range_wc = st.slider("Word Count", wc_min, wc_max, (wc_min, wc_max), step=50, label_visibility="collapsed")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Informasi Dataset</p>', unsafe_allow_html=True)
    st.caption(f"File: `{os.path.basename(data_path)}`")
    st.caption(f"Dimensi: {df.shape[0]:,} baris × {df.shape[1]} kolom")
    st.caption(f"Kategori: {df['Category'].nunique()} jabatan")
    st.caption(f"Role group: {df['Role_Group'].nunique()} kelompok")

if not kategori_dipilih:
    kategori_dipilih = cat_list
if not group_dipilih:
    group_dipilih = role_groups

df_f = df[
    (df['Category'].isin(kategori_dipilih)) &
    (df['Role_Group'].isin(group_dipilih)) &
    (df['Word_Count'] >= range_wc[0]) &
    (df['Word_Count'] <= range_wc[1])
].copy()

if df_f.empty:
    st.warning("Tidak ada data sesuai kombinasi filter. Longgarkan filter kategori/word count.")
    st.stop()


# ==============================================================
# HEADER
# ==============================================================

st.markdown('<p class="page-title">CV Classification Dashboard</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="page-subtitle">Analisis dataset CV multi-role dengan {df["Category"].nunique()} kategori jabatan — termasuk AI, ML, Front End, Back End, Full-Stack, Mobile, dan Project Management.</p>',
    unsafe_allow_html=True
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ==============================================================
# METRIC CARDS
# ==============================================================

critical_available = sum(cat in set(df['Category']) for cat in CRITICAL_CATEGORIES)
rare_categories = (df['Category'].value_counts() < 100).sum()

cols = st.columns(5)
metrics = [
    (f"{len(df_f):,}", "Total CV", f"dari {len(df):,} data"),
    (f"{df_f['Category'].nunique()}", "Kategori", "jabatan terfilter"),
    (f"{critical_available}/{len(CRITICAL_CATEGORIES)}", "Kategori Krusial", "role baru tersedia"),
    (f"{df_f['Word_Count'].mean():.0f}", "Rata-rata Kata", "per CV"),
    (f"{rare_categories}", "Kategori <100", "perlu enrichment"),
]
for col, (val, label, sub) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="card">
            <p class="card-value">{val}</p>
            <p class="card-label">{label}</p>
            <p class="card-sub">{sub}</p>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================
# TABS
# ==============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Distribusi Kategori",
    "Role Coverage",
    "Analisis Word Count",
    "Word Cloud",
    "Perbandingan & Outlier",
    "Dataset",
])


# ── TAB 1 — DISTRIBUSI ──────────────────────────────────────────
with tab1:
    cat_counts = df_f['Category'].value_counts().reset_index()
    cat_counts.columns = ['Category', 'Jumlah']

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-header">Jumlah CV per Kategori</p>', unsafe_allow_html=True)
        fig = px.bar(
            cat_counts.sort_values('Jumlah'),
            x='Jumlah', y='Category', orientation='h',
            color='Category', text='Jumlah',
            color_discrete_sequence=COLORS,
        )
        fig.update_traces(textposition='outside', textfont_size=11, cliponaxis=False)
        fig.update_layout(**LAYOUT)
        fig.update_layout(height=max(380, 32 * len(cat_counts)), showlegend=False,
                          xaxis_title='Jumlah CV', yaxis_title='',
                          margin=dict(t=40, b=40, l=190, r=60))
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with col_b:
        st.markdown('<p class="section-header">Proporsi per Kategori</p>', unsafe_allow_html=True)
        fig = px.pie(
            cat_counts, names='Category', values='Jumlah',
            color_discrete_sequence=COLORS, hole=0.45,
        )
        fig.update_traces(textposition='inside', textinfo='percent',
                          hovertemplate='<b>%{label}</b><br>%{value} CV (%{percent})<extra></extra>')
        fig.update_layout(**LAYOUT, height=max(380, 32 * len(cat_counts)), showlegend=True,
                          legend=dict(orientation='v', x=1.02, y=0.5, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown('<p class="section-header">Ringkasan Statistik per Kategori</p>', unsafe_allow_html=True)
    summary = df_f.groupby('Category').agg(
        Jumlah_CV=('ID_CV','count'),
        Role_Group=('Role_Group', lambda x: x.mode().iloc[0] if not x.mode().empty else '-'),
        Rata_rata_Kata=('Word_Count','mean'),
        Median_Kata=('Word_Count','median'),
        Min_Kata=('Word_Count','min'),
        Maks_Kata=('Word_Count','max'),
        Std_Dev=('Word_Count','std'),
    ).round(1).sort_values('Jumlah_CV', ascending=False).reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ── TAB 2 — ROLE COVERAGE ───────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Coverage Kategori Krusial yang Ditambahkan</p>', unsafe_allow_html=True)
    critical_df = pd.DataFrame({'Category': CRITICAL_CATEGORIES})
    critical_df['Jumlah'] = critical_df['Category'].map(df['Category'].value_counts()).fillna(0).astype(int)
    critical_df['Status'] = critical_df['Jumlah'].apply(lambda x: 'Aman' if x >= 250 else ('Perlu Enrichment' if x < 100 else 'Cukup'))

    col_a, col_b = st.columns([1.25, 1])
    with col_a:
        fig = px.bar(
            critical_df.sort_values('Jumlah'),
            x='Jumlah', y='Category', orientation='h',
            color='Status', text='Jumlah',
            color_discrete_map={'Aman': '#2e7d6e', 'Cukup': '#d4860b', 'Perlu Enrichment': '#c0392b'},
        )
        fig.update_traces(textposition='outside', cliponaxis=False)
        fig.update_layout(**LAYOUT, height=390, xaxis_title='Jumlah CV', yaxis_title='', margin=dict(t=40, b=40, l=180, r=60))
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with col_b:
        added = ', '.join([f'<span class="badge">{c}</span>' for c in CRITICAL_CATEGORIES])
        small = critical_df[critical_df['Jumlah'] < 100]['Category'].tolist()
        recommendation = (
            'Dataset sudah memuat semua kategori krusial. Kategori AI Engineer dan ML Engineer tetap perlu ditambah sampel jika ingin training model lebih stabil.'
            if small else
            'Coverage kategori krusial sudah relatif aman untuk eksplorasi awal.'
        )
        st.markdown(f"""
        <div class="insight-box">
            <b>Kategori baru:</b><br>{added}<br><br>
            <b>Catatan kualitas:</b><br>{recommendation}
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(critical_df, use_container_width=True, hide_index=True)

    st.markdown('<p class="section-header">Distribusi Role Group</p>', unsafe_allow_html=True)
    group_counts = df_f['Role_Group'].value_counts().reset_index()
    group_counts.columns = ['Role_Group', 'Jumlah']
    fig = px.bar(group_counts, x='Role_Group', y='Jumlah', text='Jumlah', color='Role_Group', color_discrete_sequence=COLORS)
    fig.update_traces(textposition='outside', cliponaxis=False)
    fig.update_layout(**LAYOUT, height=360, showlegend=False, xaxis_title='', yaxis_title='Jumlah CV')
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ── TAB 3 — WORD COUNT ──────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Distribusi Word Count</p>', unsafe_allow_html=True)
        fig = px.histogram(df_f, x='Word_Count', nbins=50, color_discrete_sequence=['#1e3a5f'])
        fig.add_vline(x=df_f['Word_Count'].mean(), line_dash='dash', line_color='#c0392b', line_width=1.5,
                      annotation_text=f"Mean {df_f['Word_Count'].mean():.0f}", annotation_font_size=11, annotation_font_color='#c0392b')
        fig.add_vline(x=df_f['Word_Count'].median(), line_dash='dot', line_color='#d4860b', line_width=1.5,
                      annotation_text=f"Median {df_f['Word_Count'].median():.0f}", annotation_font_size=11, annotation_font_color='#d4860b')
        fig.update_layout(**LAYOUT, height=350, xaxis_title='Jumlah Kata', yaxis_title='Frekuensi')
        st.plotly_chart(fig, use_container_width=True, theme=None)

    with col_b:
        st.markdown('<p class="section-header">Boxplot per Kategori</p>', unsafe_allow_html=True)
        fig = px.box(df_f, x='Category', y='Word_Count', color='Category', points='outliers', color_discrete_sequence=COLORS)
        fig.update_layout(**LAYOUT, height=350, showlegend=False, xaxis_tickangle=-30, xaxis_title='', yaxis_title='Jumlah Kata')
        st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown('<p class="section-header">Violin Plot</p>', unsafe_allow_html=True)
    fig = px.violin(df_f, x='Category', y='Word_Count', color='Category', box=True, points=False, color_discrete_sequence=COLORS)
    fig.update_layout(**LAYOUT, height=420, showlegend=False, xaxis_tickangle=-25, xaxis_title='', yaxis_title='Jumlah Kata')
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown('<p class="section-header">Korelasi Word Count vs Character Count</p>', unsafe_allow_html=True)
    fig = px.scatter(df_f, x='Word_Count', y='Char_Count', color='Category', opacity=0.45, color_discrete_sequence=COLORS, hover_data=['ID_CV', 'Category', 'Original_Category'])
    fig.update_layout(**LAYOUT, height=390, xaxis_title='Word Count', yaxis_title='Character Count', legend=dict(orientation='h', y=-0.2, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption(f"Korelasi Pearson: **{df_f['Word_Count'].corr(df_f['Char_Count']):.4f}** — kedua fitur cenderung redundan, cukup gunakan salah satu saat pemodelan.")


# ── TAB 4 — WORD CLOUD ──────────────────────────────────────────
with tab4:
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        pilihan = st.selectbox("Tampilkan untuk:", ['Semua Kategori'] + cat_list)

    subset = df_f if pilihan == 'Semua Kategori' else df_f[df_f['Category'] == pilihan]
    top_words = get_top_words(subset['Clean_Text'], n=80)

    if not top_words:
        st.warning("Tidak cukup data untuk Word Cloud.")
    else:
        wc_obj = WordCloud(
            width=1400, height=500,
            background_color=wordcloud_bg,
            colormap='Blues',
            max_words=80,
            prefer_horizontal=0.75,
            min_font_size=10,
        ).generate_from_frequencies(dict(top_words))

        fig_wc, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(wc_obj, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig_wc)
        plt.close(fig_wc)

        st.markdown('<p class="section-header">Top 20 Kata</p>', unsafe_allow_html=True)
        top20 = pd.DataFrame(top_words[:20], columns=['Kata', 'Frekuensi'])
        fig = px.bar(top20.sort_values('Frekuensi'), x='Frekuensi', y='Kata', orientation='h', color='Frekuensi', color_continuous_scale='Blues')
        fig.update_layout(**LAYOUT, height=470, showlegend=False, coloraxis_showscale=False, xaxis_title='Frekuensi', yaxis_title='')
        st.plotly_chart(fig, use_container_width=True, theme=None)


# ── TAB 5 — PERBANDINGAN & OUTLIER ──────────────────────────────
with tab5:
    st.markdown('<p class="section-header">Perbandingan Dua Kategori</p>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        cat1 = st.selectbox("Kategori A", cat_list, index=0)
    with col_b:
        cat2 = st.selectbox("Kategori B", cat_list, index=min(1, len(cat_list)-1))

    if cat1 == cat2:
        st.info("Pilih dua kategori yang berbeda.")
    else:
        df_comp = df_f[df_f['Category'].isin([cat1, cat2])]
        fig = px.histogram(df_comp, x='Word_Count', color='Category', barmode='overlay', nbins=40, opacity=0.7, color_discrete_sequence=['#1e3a5f','#c0392b'])
        fig.update_layout(**LAYOUT, height=350, xaxis_title='Jumlah Kata', yaxis_title='Frekuensi', legend=dict(orientation='h', y=1.08))
        st.plotly_chart(fig, use_container_width=True, theme=None)

        c1, c2 = st.columns(2)
        for col_ui, cat in zip([c1, c2], [cat1, cat2]):
            s = df_f[df_f['Category'] == cat]['Word_Count']
            with col_ui:
                st.markdown(f"**{cat}**")
                st.markdown(f"""
                <div class="card" style="text-align:left; padding:14px 18px;">
                    <table style="width:100%; font-size:0.83rem; border-collapse:collapse;">
                        <tr><td style="padding:4px 0; color:{text_sub};">Jumlah CV</td><td style="font-weight:600; text-align:right;">{len(s):,}</td></tr>
                        <tr><td style="padding:4px 0; color:{text_sub};">Rata-rata</td><td style="font-weight:600; text-align:right;">{s.mean():.0f} kata</td></tr>
                        <tr><td style="padding:4px 0; color:{text_sub};">Median</td><td style="font-weight:600; text-align:right;">{s.median():.0f} kata</td></tr>
                        <tr><td style="padding:4px 0; color:{text_sub};">Std Deviasi</td><td style="font-weight:600; text-align:right;">{s.std():.0f} kata</td></tr>
                        <tr><td style="padding:4px 0; color:{text_sub};">Minimum</td><td style="font-weight:600; text-align:right;">{s.min()} kata</td></tr>
                        <tr><td style="padding:4px 0; color:{text_sub};">Maksimum</td><td style="font-weight:600; text-align:right;">{s.max()} kata</td></tr>
                    </table>
                </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Deteksi Outlier — Metode IQR</p>', unsafe_allow_html=True)
    q1 = df_f['Word_Count'].quantile(0.25)
    q3 = df_f['Word_Count'].quantile(0.75)
    iqr = q3 - q1
    batas_bawah = q1 - 1.5 * iqr
    batas_atas  = q3 + 1.5 * iqr
    df_out = df_f.copy()
    df_out['Status'] = df_out['Word_Count'].apply(lambda x: 'Outlier' if x < batas_bawah or x > batas_atas else 'Normal')

    fig = px.scatter(
        df_out, x=df_out.index, y='Word_Count',
        color='Status',
        color_discrete_map={'Normal': '#1e3a5f', 'Outlier': '#c0392b'},
        hover_data=['ID_CV', 'Category', 'Word_Count'],
        opacity=0.55,
    )
    fig.add_hline(y=batas_atas, line_dash='dash', line_color='#c0392b', line_width=1, annotation_text=f"Batas atas: {batas_atas:.0f}", annotation_font_size=11)
    fig.add_hline(y=batas_bawah, line_dash='dash', line_color='#d4860b', line_width=1, annotation_text=f"Batas bawah: {batas_bawah:.0f}", annotation_font_size=11)
    fig.update_layout(**LAYOUT, height=370, xaxis_title='Index CV', yaxis_title='Jumlah Kata', legend=dict(orientation='h', y=1.08))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    n_out = (df_out['Status'] == 'Outlier').sum()
    st.caption(f"Ditemukan **{n_out} outlier** dari {len(df_out):,} CV ({n_out/len(df_out)*100:.1f}%) — outlier dipertahankan karena konten teks tetap valid.")

    if n_out > 0:
        with st.expander("Lihat detail CV outlier"):
            st.dataframe(
                df_out[df_out['Status'] == 'Outlier'][['ID_CV','Category','Original_Category','Word_Count']]
                .sort_values('Word_Count', ascending=False),
                use_container_width=True, hide_index=True
            )


# ── TAB 6 — DATASET ─────────────────────────────────────────────
with tab6:
    st.markdown('<p class="section-header">Pencarian & Filter</p>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        keyword = st.text_input("Cari kata kunci di teks CV", placeholder="Contoh: python, machine learning, sql, react native...")
    with col_b:
        filter_cat = st.selectbox("Filter Kategori", ['Semua'] + cat_list)
    with col_c:
        filter_group = st.selectbox("Filter Role Group", ['Semua'] + role_groups)

    df_view = df_f.copy()
    if keyword:
        df_view = df_view[df_view['Clean_Text'].str.contains(keyword, case=False, na=False) | df_view['Text'].str.contains(keyword, case=False, na=False)]
    if filter_cat != 'Semua':
        df_view = df_view[df_view['Category'] == filter_cat]
    if filter_group != 'Semua':
        df_view = df_view[df_view['Role_Group'] == filter_group]

    st.caption(f"Menampilkan **{len(df_view):,}** dari {len(df_f):,} CV")

    display_cols = ['ID_CV', 'Category', 'Original_Category', 'Role_Group', 'Word_Count', 'Clean_Text']
    tampil = df_view[display_cols].copy()
    tampil['Clean_Text'] = tampil['Clean_Text'].str[:180] + '...'
    st.dataframe(tampil, use_container_width=True, hide_index=True, height=430)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    csv = df_view[['ID_CV','Category','Original_Category','Role_Group','Word_Count','Char_Count','Clean_Text','Text']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Dataset Terfilter (CSV)",
        data=csv,
        file_name='Dataset_CV_filtered.csv',
        mime='text/csv',
    )
