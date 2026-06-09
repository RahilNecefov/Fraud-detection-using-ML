# python -m streamlit run fraud_app.py

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection AI",
    page_icon="🛡️",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }

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
.badge .dot { width: 6px; height: 6px; border-radius: 50%; background: #00d4ff; flex-shrink: 0; }
.badge.highlight { border-color: #00d4ff33; color: #00d4ff; }
.badge.advisor   { border-color: #7c3aed55; color: #a78bfa; }
.badge.advisor .dot { background: #a78bfa; }

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
.result-fraud {
    background: linear-gradient(135deg, #ff4444, #cc0000);
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    margin-top: 1rem;
}
.result-legit {
    background: linear-gradient(135deg, #00c853, #007d33);
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    margin-top: 1rem;
}
.reason-box {
    background: #1a0a0a;
    border: 1px solid #3a1a1a;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-top: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #ff9999;
    line-height: 1.8;
}
.reason-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #ff4444;
    margin-bottom: 0.5rem;
}
.summary-box {
    background: #131929;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.9;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00d4ff, #0073e6);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    border: none;
    padding: 0.75rem;
    border-radius: 8px;
    cursor: pointer;
    margin-top: 1rem;
}
.stButton > button:hover { background: linear-gradient(135deg, #0099cc, #0055b3); }
label, .stSelectbox label, .stNumberInput label {
    color: #94a3b8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #131929 !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 6px !important;
}
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


# ── Rule-based fraud detection ─────────────────────────────────────────────────
def check_fraud(transaction_type, amount, ob_org, nb_org, ob_dest, nb_dest, tol=0.01):
    """
    Pure if/else rule engine covering all 4 transaction types.
    Returns (is_fraud: bool, reasons: list[str]).
    tol = floating-point tolerance to avoid false positives from rounding.

    Column meanings:
      ob_org  = oldbalanceOrg  (origin balance BEFORE)
      nb_org  = newbalanceOrig (origin balance AFTER)
      ob_dest = oldbalanceDest (destination balance BEFORE)
      nb_dest = newbalanceDest (destination balance AFTER)
    """
    reasons = []

    # ── Universal rules (every type) ───────────────────────────────────────
    if amount <= 0:
        reasons.append("Amount is zero or negative — not a valid transaction.")

    if ob_org < 0 or nb_org < 0 or ob_dest < 0 or nb_dest < 0:
        reasons.append("A balance is negative — impossible account state.")

    # ── TRANSFER ───────────────────────────────────────────────────────────
    # Money moves from one account to ANOTHER tracked account.
    # Both sides must reconcile exactly.
    #
    # Legitimate example:
    #   amount=100, ob_org=500 → nb_org=400, ob_dest=200 → nb_dest=300  ✓
    #
    # Fraud examples:
    #   origin not reduced          → ob_org=500, nb_org=500             ✗
    #   origin reduced too much     → ob_org=500, nb_org=300 (took 200)  ✗
    #   dest didn't receive funds   → ob_dest=200, nb_dest=200           ✗
    #   dest received too much      → ob_dest=200, nb_dest=400           ✗
    #   origin wiped (account drain)→ ob_org=500, amount=100, nb_org=0   ✗
    # ──────────────────────────────────────────────────────────────────────
    if transaction_type == "TRANSFER":

        # 1. Insufficient funds
        if amount > ob_org + tol:
            reasons.append("TRANSFER: amount exceeds origin balance — insufficient funds.")

        # 2. Origin balance must go DOWN
        if nb_org > ob_org + tol:
            reasons.append("TRANSFER: origin balance increased — impossible for a transfer.")

        # 3. Origin must not drop by MORE than the amount
        if nb_org < (ob_org - amount) - tol:
            reasons.append("TRANSFER: origin balance dropped more than the transfer amount — extra funds taken.")

        # 4. Origin final balance must equal ob_org - amount exactly
        if abs(nb_org - (ob_org - amount)) > tol and amount <= ob_org + tol:
            reasons.append("TRANSFER: origin balance after transfer doesn't match expected (ob_org - amount).")

        # 5. Destination must go UP
        if nb_dest < ob_dest - tol:
            reasons.append("TRANSFER: destination balance decreased — funds disappeared in transit.")

        # 6. Destination must receive AT LEAST the amount
        if nb_dest < (ob_dest + amount) - tol:
            reasons.append("TRANSFER: destination did not receive the full transfer amount.")

        # 7. Destination must not receive MORE than the amount
        if nb_dest > (ob_dest + amount) + tol:
            reasons.append("TRANSFER: destination received more than the transfer amount — fund inflation.")

        # 8. Destination final balance must equal ob_dest + amount exactly
        if abs(nb_dest - (ob_dest + amount)) > tol:
            reasons.append("TRANSFER: destination balance after transfer doesn't match expected (ob_dest + amount).")

        # 9. Account-drain pattern: origin wiped to zero but amount < full balance
        if nb_org == 0 and ob_org > amount + tol:
            reasons.append("TRANSFER: origin drained to zero despite only a partial transfer — account-wipe pattern.")

        # 10. Zero-amount transfer that still changed balances
        if amount <= tol and (abs(nb_org - ob_org) > tol or abs(nb_dest - ob_dest) > tol):
            reasons.append("TRANSFER: zero-amount transfer caused balance changes — suspicious.")

    # ── CASH_OUT ───────────────────────────────────────────────────────────
    # Origin withdraws cash via an agent. Origin must decrease.
    # Destination is the cash agent — they hand out cash so their balance
    # can decrease (they lose cash) or stay the same. It must NOT increase
    # (agent can't gain money by giving cash out).
    #
    # Legitimate examples:
    #   amount=100, ob_org=500 → nb_org=400, ob_dest=300 → nb_dest=200  ✓ (agent paid out)
    #   amount=100, ob_org=500 → nb_org=400, ob_dest=300 → nb_dest=300  ✓ (agent balance unchanged)
    #
    # Fraud examples:
    #   origin not reduced          → nb_org >= ob_org                   ✗
    #   origin reduced too much     → nb_org < ob_org - amount           ✗
    #   agent balance increased     → nb_dest > ob_dest                  ✗
    #   origin wiped (account drain)→ ob_org=500, amount=100, nb_org=0   ✗
    # ──────────────────────────────────────────────────────────────────────
    elif transaction_type == "CASH_OUT":

        # 1. Insufficient funds
        if amount > ob_org + tol:
            reasons.append("CASH_OUT: amount exceeds origin balance — insufficient funds.")

        # 2. Origin balance must go DOWN
        if nb_org >= ob_org - tol and amount > tol:
            reasons.append("CASH_OUT: origin balance didn't decrease after cash-out.")

        # 3. Origin must not drop by MORE than the amount
        if nb_org < (ob_org - amount) - tol:
            reasons.append("CASH_OUT: origin over-decremented — more taken than the cash-out amount.")

        # 4. Origin final balance must equal ob_org - amount exactly
        if abs(nb_org - (ob_org - amount)) > tol and amount <= ob_org + tol:
            reasons.append("CASH_OUT: origin balance after cash-out doesn't match expected (ob_org - amount).")

        # 5. Account-drain pattern
        if nb_org == 0 and ob_org > amount + tol:
            reasons.append("CASH_OUT: origin drained to zero despite only a partial cash-out — account-wipe pattern.")

        # 6. Zero cash-out that still changed origin balance
        if amount <= tol and abs(nb_org - ob_org) > tol:
            reasons.append("CASH_OUT: zero-amount cash-out caused origin balance change — suspicious.")

    # ── CASH_IN ────────────────────────────────────────────────────────────
    # Cash deposited INTO a destination account via an agent.
    # Only the destination balance is checked — origin is not a tracked
    # bank account (it represents external cash, not another account).
    #
    # Legitimate example:
    #   amount=100, ob_dest=200 → nb_dest=300  ✓
    #
    # Fraud examples:
    #   dest didn't increase        → nb_dest <= ob_dest                 ✗
    #   dest received wrong amount  → nb_dest != ob_dest + amount        ✗
    #   dest received too much      → nb_dest > ob_dest + amount         ✗
    # ──────────────────────────────────────────────────────────────────────
    elif transaction_type == "CASH_IN":

        # 1. Destination balance must go UP
        if nb_dest <= ob_dest - tol and amount > tol:
            reasons.append("CASH_IN: destination balance didn't increase after deposit.")

        # 2. Destination must receive at least the amount
        if nb_dest < (ob_dest + amount) - tol:
            reasons.append("CASH_IN: destination received less than the deposited amount.")

        # 3. Destination must not receive more than the amount
        if nb_dest > (ob_dest + amount) + tol:
            reasons.append("CASH_IN: destination received more than the deposited amount — fund inflation.")

        # 4. Destination final balance must equal ob_dest + amount exactly
        if abs(nb_dest - (ob_dest + amount)) > tol:
            reasons.append("CASH_IN: destination balance after deposit doesn't match expected (ob_dest + amount).")

        # 5. Zero deposit that still changed destination balance
        if amount <= tol and abs(nb_dest - ob_dest) > tol:
            reasons.append("CASH_IN: zero-amount deposit caused destination balance change — suspicious.")

    # ── PAYMENT ────────────────────────────────────────────────────────────
    # Payer sends money to a merchant. The merchant's account in this
    # dataset is typically not tracked (always shows 0/0), so we only
    # enforce origin-side rules. Destination must not change.
    #
    # Legitimate example:
    #   amount=100, ob_org=500 → nb_org=400, ob_dest=0 → nb_dest=0  ✓
    #
    # Fraud examples:
    #   payer not debited           → nb_org >= ob_org                   ✗
    #   payer debited too much      → nb_org < ob_org - amount           ✗
    #   destination changed at all  → nb_dest != ob_dest                 ✗
    #   payer wiped (account drain) → ob_org=500, amount=100, nb_org=0   ✗
    # ──────────────────────────────────────────────────────────────────────
    elif transaction_type == "PAYMENT":

        # 1. Insufficient funds
        if amount > ob_org + tol:
            reasons.append("PAYMENT: amount exceeds payer balance — insufficient funds.")

        # 2. Payer balance must go DOWN
        if nb_org >= ob_org - tol and amount > tol:
            reasons.append("PAYMENT: payer balance didn't decrease after payment.")

        # 3. Payer must not be over-debited
        if nb_org < (ob_org - amount) - tol:
            reasons.append("PAYMENT: payer over-debited — more deducted than the payment amount.")

        # 4. Payer final balance must equal ob_org - amount exactly
        if abs(nb_org - (ob_org - amount)) > tol and amount <= ob_org + tol:
            reasons.append("PAYMENT: payer balance after payment doesn't match expected (ob_org - amount).")

        # 5. Account-drain pattern
        if nb_org == 0 and ob_org > amount + tol:
            reasons.append("PAYMENT: payer balance wiped to zero despite partial payment — account-wipe pattern.")

        # 6. Zero payment that still changed payer balance
        if amount <= tol and abs(nb_org - ob_org) > tol:
            reasons.append("PAYMENT: zero-amount payment caused payer balance change — suspicious.")

    is_fraud = len(reasons) > 0
    return is_fraud, reasons


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <div class="main-title">🛡️ Fraud Detection AI</div>
    <div class="sub-title">Rule-based financial transaction analysis · Logic engine</div>
    <div class="author-strip">
        <div class="badge highlight"><span class="dot"></span>Rahil Najafov</div>
        <div class="badge"><span class="dot"></span>Baku Higher Oil School</div>
        <div class="badge"><span class="dot"></span>Process Automation Engineering</div>
        <div class="badge advisor"><span class="dot"></span>Advisor — Associate prof. Leyla Muradkhanli</div>
        <div class="badge"><span class="dot"></span>Graduation Project · 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Input form ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)
    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_IN", "CASH_OUT", "PAYMENT", "TRANSFER"]
    )
    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0, max_value=10_000_000.0,
        value=1000.0, step=100.0, format="%.2f"
    )

    st.markdown('<div class="section-header">Origin Account</div>', unsafe_allow_html=True)
    oldbalanceOrg = st.number_input(
        "Balance Before Transaction ($)",
        min_value=0.0, max_value=100_000_000.0,
        value=5000.0, step=100.0, format="%.2f"
    )
    newbalanceOrig = st.number_input(
        "Balance After Transaction ($)",
        min_value=0.0, max_value=100_000_000.0,
        value=4000.0, step=100.0, format="%.2f"
    )

with col2:
    st.markdown('<div class="section-header">Destination Account</div>', unsafe_allow_html=True)
    oldbalanceDest = st.number_input(
        "Dest. Balance Before ($)",
        min_value=0.0, max_value=100_000_000.0,
        value=2000.0, step=100.0, format="%.2f"
    )
    newbalanceDest = st.number_input(
        "Dest. Balance After ($)",
        min_value=0.0, max_value=100_000_000.0,
        value=3000.0, step=100.0, format="%.2f"
    )

    st.markdown('<div class="section-header">Run Analysis</div>', unsafe_allow_html=True)

    if st.button("🔍 Analyze Transaction"):
        is_fraud, reasons = check_fraud(
            transaction_type,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest,
        )

        if is_fraud:
            st.markdown(
                '<div class="result-fraud">⚠️ FRAUDULENT TRANSACTION DETECTED</div>',
                unsafe_allow_html=True
            )
            reasons_html = "".join(f"▸ {r}<br>" for r in reasons)
            st.markdown(
                f'<div class="reason-title">Triggered Rules</div>'
                f'<div class="reason-box">{reasons_html}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-legit">✅ TRANSACTION APPEARS LEGITIMATE</div>',
                unsafe_allow_html=True
            )

        # ── Balance reconciliation summary ──
        if transaction_type == "CASH_IN":
            exp_orig_display = "N/A (external cash — not a tracked account)"
            exp_dest_display = f"${oldbalanceDest + amount:,.2f}"
        elif transaction_type == "CASH_OUT":
            exp_orig_display = f"${oldbalanceOrg - amount:,.2f}"
            exp_dest_display = "N/A (agent side — not tracked)"
        elif transaction_type == "TRANSFER":
            exp_orig_display = f"${oldbalanceOrg - amount:,.2f}"
            exp_dest_display = f"${oldbalanceDest + amount:,.2f}"
        elif transaction_type == "PAYMENT":
            exp_orig_display = f"${oldbalanceOrg - amount:,.2f}"
            exp_dest_display = f"${oldbalanceDest:,.2f} (unchanged)"

        st.markdown(f"""
        <div class="summary-box">
            TYPE &nbsp;→&nbsp; {transaction_type}<br>
            AMOUNT &nbsp;→&nbsp; ${amount:,.2f}<br>
            <br>
            ORIGIN BEFORE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→&nbsp; ${oldbalanceOrg:,.2f}<br>
            ORIGIN AFTER (actual) &nbsp;&nbsp;→&nbsp; ${newbalanceOrig:,.2f}<br>
            ORIGIN AFTER (expected) →&nbsp; {exp_orig_display}<br>
            <br>
            DEST BEFORE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→&nbsp; ${oldbalanceDest:,.2f}<br>
            DEST AFTER (actual) &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp; ${newbalanceDest:,.2f}<br>
            DEST AFTER (expected) &nbsp;&nbsp;→&nbsp; {exp_dest_display}<br>
            <br>
            RULES TRIGGERED &nbsp;→&nbsp; {len(reasons)}
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Rahil Najafov · Baku Higher Oil School · Process Automation Engineering · Advisor: Associate prof. Leyla Muradkhanli
</div>
""", unsafe_allow_html=True)
