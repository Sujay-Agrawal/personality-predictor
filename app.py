import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Personality Predictor", page_icon="🧠", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.1rem; }
    .hero p  { margin: .3rem 0 0 0; opacity: .92; font-size: 1.05rem; }
    .result-card {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(16,185,129,.25);
    }
    .result-card .label { font-size: .95rem; letter-spacing: .12em; text-transform: uppercase; opacity: .9; }
    .result-card .value { font-size: 2.6rem; font-weight: 800; margin-top: .2rem; }
    .result-card .conf  { font-size: 1rem; opacity: .95; margin-top: .2rem; }
    div.stButton > button {
        border-radius: 12px; padding: .7rem 1rem; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load files ----------
@st.cache_resource
def load_files():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    enc_path = os.path.join(BASE_DIR, "label_encoder.pkl")
    encoder = joblib.load(enc_path) if os.path.exists(enc_path) else None
    return model, scaler, encoder


model, scaler, label_encoder = load_files()

# ---------- Feature definitions (the 26 training columns) ----------
SECTIONS = {
    "🗣️ Social": [
        "social_energy", "alone_time_preference", "talkativeness", "group_comfort",
        "party_liking", "public_speaking_comfort", "friendliness", "online_social_usage",
    ],
    "🧩 Thinking & Work": [
        "deep_reflection", "listening_skill", "empathy", "organization", "leadership",
        "curiosity", "planning", "decision_speed", "work_style_collaborative",
    ],
    "🎒 Lifestyle & Interests": [
        "routine_preference", "spontaneity", "risk_taking", "excitement_seeking",
        "adventurousness", "reading_habit", "sports_interest", "travel_desire", "gadget_usage",
    ],
}
ALL_FEATURES = [c for cols in SECTIONS.values() for c in cols]

# Exact column order used when training (X = df.drop(...))
TRAINING_ORDER = [
    "social_energy", "alone_time_preference", "talkativeness", "deep_reflection",
    "group_comfort", "party_liking", "listening_skill", "empathy", "organization",
    "leadership", "risk_taking", "public_speaking_comfort", "curiosity",
    "routine_preference", "excitement_seeking", "friendliness", "planning",
    "spontaneity", "adventurousness", "reading_habit", "sports_interest",
    "online_social_usage", "travel_desire", "gadget_usage",
    "work_style_collaborative", "decision_speed",
]

# What each model output (0, 1, 2) means. EDIT THESE to match how you encoded the target.
LABEL_MAP = {
    0: "Introvert",
    1: "Ambivert",
    2: "Extrovert",
}


def label_name(value):
    """Turn a model output (0/1/2) into a readable personality name."""
    if label_encoder is not None:
        return str(label_encoder.inverse_transform([value])[0])
    try:
        return LABEL_MAP.get(int(value), str(value))
    except (ValueError, TypeError):
        return str(value)


# Change these if your data uses a different scale
MIN_VAL, MAX_VAL, DEFAULT_VAL = 0, 10, 5

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app predicts a personality type from **26 self-rated traits** "
        "using a Logistic Regression model."
    )
    st.write(f"Rate each trait from **{MIN_VAL}** (low) to **{MAX_VAL}** (high).")
    st.divider()
    if st.button("🔄 Reset all sliders", use_container_width=True):
        for f in ALL_FEATURES:
            st.session_state[f] = DEFAULT_VAL
        st.rerun()

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 Personality Type Predictor</h1>
        <p>Rate yourself on every trait below, then hit <b>Predict</b> at the bottom to see your result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Inputs ----------
inputs = {}
for section, columns in SECTIONS.items():
    st.subheader(section)
    cols = st.columns(3)
    for i, name in enumerate(columns):
        label = name.replace("_", " ").title()
        inputs[name] = cols[i % 3].slider(
            label, MIN_VAL, MAX_VAL, DEFAULT_VAL, key=name
        )
    st.divider()

st.write("")
predict_clicked = st.button("🔮 Predict my personality", type="primary", use_container_width=True)

# ---------- Prediction ----------
if predict_clicked:
    # Use the order the scaler/model saw during fit; fall back to the training order
    if hasattr(scaler, "feature_names_in_"):
        feature_order = list(scaler.feature_names_in_)
    elif hasattr(model, "feature_names_in_"):
        feature_order = list(model.feature_names_in_)
    else:
        feature_order = TRAINING_ORDER

    missing = [c for c in feature_order if c not in inputs]
    if missing:
        st.error(f"The scaler/model expects columns that are not in the form: {missing}")
        st.stop()

    input_df = pd.DataFrame([inputs])[feature_order]
    scaled = scaler.transform(input_df)

    raw_pred = model.predict(scaled)[0]
    pred = label_name(raw_pred)

    confidence_text = ""
    proba_df = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
        classes = [label_name(c) for c in model.classes_]
        proba_df = pd.DataFrame({"Type": classes, "Probability": proba})
        proba_df = proba_df.sort_values("Probability", ascending=False).reset_index(drop=True)
        confidence_text = f"Confidence: {proba_df.loc[0, 'Probability'] * 100:.1f}%"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">Your predicted personality type</div>
            <div class="value">{pred}</div>
            <div class="conf">{confidence_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if proba_df is not None:
        st.subheader("Prediction breakdown")
        for _, row in proba_df.iterrows():
            st.write(f"**{row['Type']}** — {row['Probability'] * 100:.1f}%")
            st.progress(float(row["Probability"]))

    with st.expander("See the values you entered"):
        st.dataframe(input_df.T.rename(columns={0: "Your rating"}), use_container_width=True)
