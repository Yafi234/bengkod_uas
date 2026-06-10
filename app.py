import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }

    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(26, 35, 126, 0.4);
    }
    .header-card h1 { color: #ffffff; font-size: 2rem; margin: 0; font-weight: 700; }
    .header-card p  { color: #90caf9; margin: 0.3rem 0 0; font-size: 1rem; }

    /* Metric cards */
    .metric-card {
        background: #1e2532;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); border-color: #4a90d9; }
    .metric-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin-top: 4px; }

    /* Result boxes */
    .result-churn {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #f87171;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
    }
    .result-no-churn {
        background: linear-gradient(135deg, #14532d, #166534);
        border: 1px solid #4ade80;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(74, 222, 128, 0.3);
    }
    .result-icon  { font-size: 4rem; margin-bottom: 0.5rem; }
    .result-title { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
    .result-sub   { color: #cbd5e1; font-size: 0.95rem; margin-top: 0.5rem; }

    /* Section headers */
    .section-header {
        color: #93c5fd;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #2d3748;
    }

    /* Probability bar */
    .prob-bar-bg {
        background: #2d3748;
        border-radius: 999px;
        height: 12px;
        width: 100%;
        margin-top: 8px;
    }
    .prob-bar-fill {
        height: 12px;
        border-radius: 999px;
        transition: width 0.5s;
    }

    /* Feature description */
    .feat-desc {
        background: #1a1f2e;
        border-left: 3px solid #4a90d9;
        padding: 0.5rem 0.8rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.82rem;
        color: #94a3b8;
        margin: 0.2rem 0 0.8rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model    = joblib.load('best_model.pkl')
        scaler   = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        return model, scaler, features, True
    except Exception as e:
        return None, None, None, False

model, scaler, features, model_loaded = load_artifacts()


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📌 Navigasi")
    page = st.radio("", ["🔍 Prediksi Churn", "📈 Tentang Model", "📚 Panduan Fitur"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🏫 Info Proyek")
    st.markdown("""
    **Program:** Bengkel Koding Data Science  
    **Universitas:** Dian Nuswantoro  
    **Dataset:** Sales & Marketing Customer  
    **Records:** 15.000  
    **Target:** Prediksi Customer Churn
    """)

    st.markdown("---")
    if model_loaded:
        st.success("✅ Model berhasil dimuat")
        st.markdown(f"**Fitur dipakai:** {len(features)}")
    else:
        st.error("❌ Model belum dimuat")
        st.info("Jalankan notebook.ipynb terlebih dahulu untuk menghasilkan best_model.pkl, scaler.pkl, features.pkl")


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>📊 Customer Churn Predictor</h1>
    <p>Sistem prediksi churn pelanggan berbasis Machine Learning — UAS Bengkel Koding Data Science, UDINUS</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 1: PREDIKSI CHURN
# ══════════════════════════════════════════════════════════════════
if page == "🔍 Prediksi Churn":

    if not model_loaded:
        st.error("⚠️ Model belum tersedia. Jalankan `notebook_uas.ipynb` dulu untuk menyimpan `best_model.pkl`, `scaler.pkl`, dan `features.pkl`.")
        st.stop()

    st.markdown("### Masukkan Data Pelanggan")
    st.markdown("Isi form di bawah untuk memprediksi apakah pelanggan berpotensi churn.")

    # ── Definisi semua fitur yang mungkin muncul ───────────────────
    FEATURE_DEFAULTS = {
        'age':                      (18.0, 80.0, 35.0),
        'is_premium_user':          (0, 1, 0),
        'total_visits':             (0, 500, 50),
        'avg_session_time':         (0.0, 120.0, 15.0),
        'pages_per_session':        (1.0, 30.0, 5.0),
        'email_open_rate':          (0.0, 1.0, 0.3),
        'email_click_rate':         (0.0, 1.0, 0.1),
        'total_spent':              (0.0, 50000.0, 500.0),
        'avg_order_value':          (0.0, 5000.0, 100.0),
        'discount_used':            (0, 1, 0),
        'support_tickets':          (0, 20, 1),
        'refund_requested':         (0, 1, 0),
        'delivery_delay_days':      (0, 30, 2),
        'satisfaction_score':       (1.0, 5.0, 3.5),
        'nps_score':                (-100, 100, 30),
        'marketing_spend_per_user': (0.0, 1000.0, 50.0),
        'lifetime_value':           (0.0, 100000.0, 1000.0),
        'last_3_month_purchase_freq': (0, 30, 3),
    }

    FEATURE_LABELS = {
        'age': 'Usia Pelanggan',
        'is_premium_user': 'Pengguna Premium (0=Tidak, 1=Ya)',
        'total_visits': 'Total Kunjungan',
        'avg_session_time': 'Rata-rata Waktu Sesi (menit)',
        'pages_per_session': 'Halaman per Sesi',
        'email_open_rate': 'Email Open Rate (0-1)',
        'email_click_rate': 'Email Click Rate (0-1)',
        'total_spent': 'Total Pengeluaran (Rp)',
        'avg_order_value': 'Rata-rata Nilai Order (Rp)',
        'discount_used': 'Penggunaan Diskon (0/1)',
        'support_tickets': 'Jumlah Tiket Support',
        'refund_requested': 'Permintaan Refund (0/1)',
        'delivery_delay_days': 'Keterlambatan Pengiriman (hari)',
        'satisfaction_score': 'Skor Kepuasan (1-5)',
        'nps_score': 'NPS Score',
        'marketing_spend_per_user': 'Biaya Marketing per User',
        'lifetime_value': 'Lifetime Value',
        'last_3_month_purchase_freq': 'Frekuensi Beli 3 Bulan Terakhir',
    }

    # ── Form Input ────────────────────────────────────────────────
    input_data = {}

    col1, col2, col3 = st.columns(3)
    feat_cols = [col1, col2, col3]

    for i, feat in enumerate(features):
        col = feat_cols[i % 3]
        label = FEATURE_LABELS.get(feat, feat.replace('_', ' ').title())

        with col:
            if feat in FEATURE_DEFAULTS:
                mn, mx, default = FEATURE_DEFAULTS[feat]
                if isinstance(default, int) and mn in [0, 1] and mx in [0, 1]:
                    input_data[feat] = st.selectbox(label, [0, 1],
                                                    index=int(default), key=feat)
                elif isinstance(default, int):
                    input_data[feat] = st.number_input(label, min_value=int(mn),
                                                        max_value=int(mx),
                                                        value=int(default), key=feat)
                else:
                    input_data[feat] = st.number_input(label, min_value=float(mn),
                                                        max_value=float(mx),
                                                        value=float(default),
                                                        step=0.01, key=feat)
            else:
                input_data[feat] = st.number_input(label, value=0.0, key=feat)

    st.markdown("---")

    # ── Tombol Prediksi ───────────────────────────────────────────
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🚀 Prediksi Sekarang", use_container_width=True, type="primary")

    if predict_btn:
        try:
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df[features])
            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0]
            prob_churn = proba[1]
            prob_no_churn = proba[0]

            st.markdown("---")
            st.markdown("### 📋 Hasil Prediksi")

            res_col1, res_col2 = st.columns([3, 2])

            with res_col1:
                if pred == 1:
                    st.markdown(f"""
                    <div class="result-churn">
                        <div class="result-icon">⚠️</div>
                        <div class="result-title">PELANGGAN CHURN</div>
                        <div class="result-sub">Pelanggan ini berpotensi tinggi untuk berhenti menggunakan layanan.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-no-churn">
                        <div class="result-icon">✅</div>
                        <div class="result-title">TIDAK CHURN</div>
                        <div class="result-sub">Pelanggan ini diprediksi akan tetap menggunakan layanan.</div>
                    </div>""", unsafe_allow_html=True)

            with res_col2:
                st.markdown("**Probabilitas Prediksi**")

                churn_pct = prob_churn * 100
                no_churn_pct = prob_no_churn * 100
                churn_color = "#ef4444" if prob_churn > 0.5 else "#f97316"
                no_churn_color = "#22c55e"

                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #ef4444; font-weight: 600;">Churn</span>
                        <span style="color: #f1f5f9; font-weight: 700;">{churn_pct:.1f}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: {churn_pct}%; background: {churn_color};"></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #22c55e; font-weight: 600;">Tidak Churn</span>
                        <span style="color: #f1f5f9; font-weight: 700;">{no_churn_pct:.1f}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width: {no_churn_pct}%; background: {no_churn_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence level
                confidence = max(prob_churn, prob_no_churn)
                if confidence >= 0.8:
                    conf_label, conf_color = "Sangat Tinggi", "#22c55e"
                elif confidence >= 0.65:
                    conf_label, conf_color = "Tinggi", "#84cc16"
                elif confidence >= 0.5:
                    conf_label, conf_color = "Sedang", "#f59e0b"
                else:
                    conf_label, conf_color = "Rendah", "#ef4444"

                st.markdown(f"""
                <div style="margin-top: 1.2rem; background: #1e2532; border-radius: 8px; padding: 0.8rem 1rem;">
                    <div style="color: #94a3b8; font-size: 0.8rem;">TINGKAT KEYAKINAN</div>
                    <div style="color: {conf_color}; font-size: 1.2rem; font-weight: 700;">{conf_label}</div>
                    <div style="color: #cbd5e1; font-size: 0.9rem;">{confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # ── Rekomendasi ───────────────────────────────────────
            st.markdown("---")
            st.markdown("### 💡 Rekomendasi Tindakan")

            if pred == 1:
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.markdown("""
                    **🎁 Retensi Pelanggan**
                    - Tawarkan diskon eksklusif
                    - Program loyalitas khusus
                    - Personal offer berdasarkan riwayat
                    """)
                with col_r2:
                    st.markdown("""
                    **📞 Engagement Aktif**
                    - Hubungi via email/telepon
                    - Survey kepuasan pelanggan
                    - Customer success check-in
                    """)
                with col_r3:
                    st.markdown("""
                    **🔧 Perbaikan Layanan**
                    - Tinjau tiket support aktif
                    - Percepat proses pengiriman
                    - Permudah proses refund
                    """)
            else:
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown("""
                    **📈 Tingkatkan Nilai**
                    - Tawarkan upgrade ke premium
                    - Cross-sell produk relevan
                    - Program referral
                    """)
                with col_r2:
                    st.markdown("""
                    **🌟 Jaga Kepuasan**
                    - Pertahankan kualitas layanan
                    - Newsletter & konten berkala
                    - Reward untuk pelanggan setia
                    """)

        except Exception as e:
            st.error(f"Error prediksi: {e}")
            st.info("Pastikan format input sudah benar dan semua file model tersedia.")


# ══════════════════════════════════════════════════════════════════
# PAGE 2: TENTANG MODEL
# ══════════════════════════════════════════════════════════════════
elif page == "📈 Tentang Model":

    st.markdown("### 🧠 Informasi Model & Eksperimen")

    # Tabel 9 model
    st.markdown("#### Rangkuman 9 Model (3 Kategori × 3 Skenario)")
    st.markdown("""
    | Skenario | Model Konvensional | Ensemble Bagging | Ensemble Voting |
    |---|---|---|---|
    | **Direct** | Logistic Regression | Random Forest | Voting (LR+SVM+KNN) |
    | **Preprocessing** | Logistic Regression | Random Forest | Voting (LR+SVM+KNN) |
    | **Hyperparameter Tuning** | Logistic Regression | Random Forest | Voting (LR+SVM+KNN) |
    """)

    st.markdown("---")
    st.markdown("#### Pipeline Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Penanganan Data:**
        - ✅ Drop kolom tidak relevan (customer_id, signup_date, last_purchase_date, coupon_code)
        - ✅ Hapus duplikasi
        - ✅ Imputasi missing value (median / modus)
        - ✅ Outlier capping dengan metode IQR
        """)
    with col2:
        st.markdown("""
        **Feature Engineering:**
        - ✅ Label Encoding untuk fitur kategorikal
        - ✅ StandardScaler (setelah train-test split)
        - ✅ Feature Selection berdasarkan Feature Importance
        - ✅ Train-Test Split 80:20 (stratified)
        """)

    st.markdown("---")
    st.markdown("#### Metrik Evaluasi yang Digunakan")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value" style="font-size:1rem; color:#93c5fd;">Proporsi prediksi benar dari total</div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-value" style="font-size:1rem; color:#93c5fd;">Ketepatan prediksi positif (churn)</div>
        </div>""", unsafe_allow_html=True)
    with mc3:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-value" style="font-size:1rem; color:#93c5fd;">Kemampuan mendeteksi semua churn</div>
        </div>""", unsafe_allow_html=True)
    with mc4:
        st.markdown("""<div class="metric-card">
            <div class="metric-label">F1-Score</div>
            <div class="metric-value" style="font-size:1rem; color:#93c5fd;">Harmonic mean precision & recall</div>
        </div>""", unsafe_allow_html=True)

    if model_loaded:
        st.markdown("---")
        st.markdown("#### Fitur yang Digunakan Model")
        feat_df = pd.DataFrame({'No': range(1, len(features)+1), 'Fitur': features})
        st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
# PAGE 3: PANDUAN FITUR
# ══════════════════════════════════════════════════════════════════
elif page == "📚 Panduan Fitur":

    st.markdown("### 📚 Panduan Lengkap Fitur Dataset")
    st.markdown("Referensi semua 30 kolom dalam dataset Sales and Marketing Customer.")

    fitur_data = [
        ("customer_id",              "int64",    "ID unik setiap pelanggan"),
        ("gender",                   "object",   "Jenis kelamin pelanggan (Male/Female)"),
        ("age",                      "float64",  "Usia pelanggan (tahun)"),
        ("country",                  "object",   "Negara asal pelanggan"),
        ("city",                     "object",   "Kota pelanggan"),
        ("signup_date",              "datetime", "Tanggal pelanggan mendaftar"),
        ("last_purchase_date",       "datetime", "Tanggal transaksi terakhir"),
        ("acquisition_channel",      "object",   "Sumber pelanggan (Email, Ads, Organic, dll)"),
        ("device_type",              "object",   "Jenis perangkat (Mobile, Desktop, Tablet)"),
        ("subscription_type",        "object",   "Jenis langganan pelanggan"),
        ("is_premium_user",          "int64",    "Status premium: 0=tidak, 1=ya"),
        ("total_visits",             "int64",    "Total kunjungan ke platform"),
        ("avg_session_time",         "float64",  "Rata-rata durasi sesi (menit)"),
        ("pages_per_session",        "float64",  "Rata-rata halaman yang dikunjungi per sesi"),
        ("email_open_rate",          "float64",  "Persentase email marketing yang dibuka (0-1)"),
        ("email_click_rate",         "float64",  "Persentase klik pada email marketing (0-1)"),
        ("total_spent",              "float64",  "Total pengeluaran pelanggan (nominal)"),
        ("avg_order_value",          "float64",  "Rata-rata nilai per transaksi"),
        ("discount_used",            "int64",    "Penggunaan diskon: 0=tidak, 1=ya"),
        ("coupon_code",              "object",   "Kode kupon yang digunakan pelanggan"),
        ("support_tickets",          "int64",    "Jumlah tiket dukungan yang dibuat"),
        ("refund_requested",         "int64",    "Permintaan refund: 0=tidak, 1=ya"),
        ("delivery_delay_days",      "int64",    "Keterlambatan pengiriman dalam hari"),
        ("payment_method",           "object",   "Metode pembayaran (Kartu Kredit, Transfer, dll)"),
        ("satisfaction_score",       "float64",  "Skor kepuasan pelanggan (1.0 - 5.0)"),
        ("nps_score",                "int64",    "Net Promoter Score (-100 hingga 100)"),
        ("marketing_spend_per_user", "float64",  "Biaya pemasaran yang dikeluarkan per pelanggan"),
        ("lifetime_value",           "float64",  "Total nilai pelanggan selama berlangganan"),
        ("last_3_month_purchase_freq","int64",   "Frekuensi pembelian dalam 3 bulan terakhir"),
        ("churn",                    "int64",    "🎯 TARGET: 0=tidak churn, 1=churn"),
    ]

    df_fitur = pd.DataFrame(fitur_data, columns=["Kolom", "Tipe", "Keterangan"])
    st.dataframe(df_fitur, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 💡 Tips Interpretasi Churn")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Indikator Risiko Churn Tinggi:**
        - 🔴 `satisfaction_score` rendah (< 3)
        - 🔴 `nps_score` negatif
        - 🔴 `support_tickets` banyak
        - 🔴 `refund_requested` = 1
        - 🔴 `last_3_month_purchase_freq` rendah
        """)
    with c2:
        st.markdown("""
        **Indikator Pelanggan Loyal:**
        - 🟢 `is_premium_user` = 1
        - 🟢 `total_spent` tinggi
        - 🟢 `lifetime_value` tinggi
        - 🟢 `email_open_rate` tinggi
        - 🟢 `total_visits` konsisten
        """)
