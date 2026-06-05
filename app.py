# python -m streamlit run fraud_app.py
 
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.simplefilter(action='ignore')
 
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="wide",
)
 
# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
 
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
 
.stApp { background: #0a0e1a; color: #e2e8f0; }
 
/* ── Header card ── */
.header-card {
    background: linear-gradient(135deg, #0d1526 0%, #111d35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.header-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00d4ff, #0073e6, #00d4ff);
}
.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #00d4ff, #0099cc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.sub-title {
    color: #64748b;
    font-size: 0.95rem;
    font-family: 'Space Mono', monospace;
    margin-bottom: 1.6rem;
}
 
/* ── Author badge strip ── */
.author-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    align-items: center;
    border-top: 1px solid #1e2d45;
    padding-top: 1.2rem;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #0a0e1a;
    border: 1px solid #1e3a5f;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    white-space: nowrap;
}
.badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00d4ff;
    flex-shrink: 0;
}
.badge.highlight { border-color: #00d4ff33; color: #00d4ff; }
.badge.advisor   { border-color: #7c3aed55; color: #a78bfa; }
.badge.advisor .dot { background: #a78bfa; }
 
/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00d4ff;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}
 
/* ── Result boxes ── */
.result-fraud {
    background: linear-gradient(135deg, #ff4444, #cc0000);
    border-radius: 12px; padding: 1.4rem;
    text-align: center; font-size: 1.5rem; font-weight: 700; color: white; margin-top: 1rem;
}
.result-legit {
    background: linear-gradient(135deg, #00c853, #007d33);
    border-radius: 12px; padding: 1.4rem;
    text-align: center; font-size: 1.5rem; font-weight: 700; color: white; margin-top: 1rem;
}
.risk-box {
    background: #131929; border: 1px solid #1e2d45;
    border-radius: 10px; padding: 1rem 1.5rem; margin-top: 1rem;
    font-family: 'Space Mono', monospace; font-size: 0.9rem; color: #94a3b8;
}
 
/* ── Inputs ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00d4ff, #0073e6);
    color: white; font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 1.1rem; border: none; padding: 0.75rem;
    border-radius: 8px; cursor: pointer; margin-top: 1rem;
}
.stButton > button:hover { background: linear-gradient(135deg, #0099cc, #0055b3); }
label, .stSelectbox label, .stNumberInput label {
    color: #94a3b8 !important; font-family: 'Space Mono', monospace !important; font-size: 0.82rem !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: #131929 !important; color: #e2e8f0 !important;
    border: 1px solid #1e2d45 !important; border-radius: 6px !important;
}
 
/* ── Footer ── */
.footer {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #334155;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #1e2d45;
}
</style>
""", unsafe_allow_html=True)
 
 
# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_fraud_model.pkl')
    except FileNotFoundError:
        st.error("❌ Model file 'xgboost_fraud_model.pkl' not found.")
        return None
 
model = load_model()
 
TYPE_MAP = {
    "CASH_IN":  [0, 0, 0, 0],
    "CASH_OUT": [1, 0, 0, 0],
    "DEBIT":    [0, 1, 0, 0],
    "PAYMENT":  [0, 0, 1, 0],
    "TRANSFER": [0, 0, 0, 1],
}
 
def preprocess_input(t, amount, ob_org, nb_org, ob_dest, nb_dest):
    return np.array([[amount, ob_org, nb_org, ob_dest, nb_dest] + TYPE_MAP[t]])
 
def get_risk_label(prob):
    if prob > 0.9:  return "🔴 VERY HIGH RISK", "#ff4444"
    elif prob > 0.7: return "🟠 HIGH RISK",      "#ff8800"
    elif prob > 0.4: return "🟡 MEDIUM RISK",     "#ffcc00"
    else:            return "🟢 LOW RISK",         "#00c853"
 
 
# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div class="main-title">🛡️ Fraud Detection AI</div>
    <div class="sub-title">Real-time financial transaction risk analysis · XGBoost classifier</div>
    <div class="author-strip">
        <div class="badge highlight"><span class="dot"></span>Rahil Najafov</div>
        <div class="badge"><span class="dot"></span>Baku Higher Oil School</div>
        <div class="badge"><span class="dot"></span>Process Automation Engineering</div>
        <div class="badge advisor"><span class="dot"></span>Advisor —  Associate prof. Leyla Muradkhanli</div>
        <div class="badge"><span class="dot"></span>Graduation Project · 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)
 
 
# ── Input form ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")
 
with col1:
    st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)
    transaction_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=10_000_000.0, value=1000.0, step=100.0, format="%.2f")
 
    st.markdown('<div class="section-header">Origin Account</div>', unsafe_allow_html=True)
    oldbalanceOrg  = st.number_input("Balance Before Transaction ($)", min_value=0.0, max_value=100_000_000.0, value=5000.0, step=100.0, format="%.2f")
    newbalanceOrig = st.number_input("Balance After Transaction ($)",  min_value=0.0, max_value=100_000_000.0, value=4000.0, step=100.0, format="%.2f")
 
with col2:
    st.markdown('<div class="section-header">Destination Account</div>', unsafe_allow_html=True)
    oldbalanceDest = st.number_input("Dest. Balance Before ($)", min_value=0.0, max_value=100_000_000.0, value=2000.0, step=100.0, format="%.2f")
    newbalanceDest = st.number_input("Dest. Balance After ($)",  min_value=0.0, max_value=100_000_000.0, value=3000.0, step=100.0, format="%.2f")
 
    st.markdown('<div class="section-header">Run Analysis</div>', unsafe_allow_html=True)
 
    if st.button("🔍 Analyze Transaction"):
        if model is None:
            st.error("Model not loaded.")
        else:
            inp = preprocess_input(transaction_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)
            prediction = model.predict(inp)[0]
            prob       = model.predict_proba(inp)[0][1]
            risk_label, risk_color = get_risk_label(prob)
 
            if prediction == 1:
                st.markdown('<div class="result-fraud">⚠️ FRAUDULENT TRANSACTION DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-legit">✅ TRANSACTION APPEARS LEGITIMATE</div>', unsafe_allow_html=True)
 
            st.markdown(f"""
            <div class="risk-box">
                <div style="margin-bottom:0.5rem;">
                    FRAUD PROBABILITY &nbsp;→&nbsp;
                    <span style="color:{risk_color}; font-weight:700;">{prob:.1%}</span>
                </div>
                <div>RISK LEVEL &nbsp;→&nbsp; <span style="color:{risk_color};">{risk_label}</span></div>
                <div style="margin-top:0.8rem; background:#0a0e1a; border-radius:6px; height:10px; overflow:hidden;">
                    <div style="width:{prob*100:.1f}%; background:linear-gradient(90deg,#00c853,#ffcc00,#ff4444); height:100%; border-radius:6px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
 
 
# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Rahil Najafov · Baku Higher Oil School · Process Automation Engineering · Advisor: Associate prof. Leyla Muradkhanli
</div>
""", unsafe_allow_html=True)
 