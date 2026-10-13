
import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_files():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


model, scaler = load_files()

# -----------------------------
# Personality Label Mapping
# Verify these labels against
# your dataset before using.
# -----------------------------
personality_mapping = {
    0: "Introvert",
    1: "Ambivert",
    2: "Extrovert"
}

# -----------------------------
# Input Features
# Must match training order
# -----------------------------
features = [
    'social_energy',
    'alone_time_preference',
    'talkativeness',
    'deep_reflection',
    'group_comfort',
    'party_liking',
    'listening_skill',
    'empathy',
    'organization',
    'leadership',
    'risk_taking',
    'public_speaking_comfort',
    'curiosity',
    'routine_preference',
    'excitement_seeking',
    'friendliness',
    'planning',
    'spontaneity',
    'adventurousness',
    'reading_habit',
    'sports_interest',
    'online_social_usage',
    'travel_desire',
    'gadget_usage',
    'work_style_collaborative',
    'decision_speed'
]

# -----------------------------
# UI
# -----------------------------
st.title("🧠 Personality Prediction")
st.write(
    "Discover your predicted personality type "
    "based on your preferences and habits."
)

st.info("Rate each feature from 1 to 10.")

# -----------------------------
# Input Form
# -----------------------------
with st.form("personality_form"):

    st.subheader("Personality Questionnaire")

    user_inputs = {}
    columns = st.columns(3)

    for i, feature in enumerate(features):
        with columns[i % 3]:
            label = feature.replace("_", " ").title()

            user_inputs[feature] = st.slider(
                label,
                min_value=1,
                max_value=10,
                value=5,
                key=feature
            )

    submitted = st.form_submit_button(
        "Predict Personality",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:

    input_df = pd.DataFrame(
        [[user_inputs[col] for col in features]],
        columns=features
    )

    try:
        # Scale the input data
        scaled_input = scaler.transform(input_df)

        # Predict the class
        prediction = model.predict(scaled_input)[0]

        # Convert numeric prediction to text
        personality_text = personality_mapping.get(
            prediction,
            personality_mapping.get(
                int(prediction),
                str(prediction)
            )
        )

        st.success("Prediction completed!")

        st.subheader("Your Predicted Personality Type")
        st.markdown(f"## 🎯 {personality_text}")

        # Display prediction probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                scaled_input
            )[0]

            probability_df = pd.DataFrame({
                "Personality Type": [
                    personality_mapping.get(
                        cls,
                        personality_mapping.get(
                            int(cls), str(cls)
                        )
                    )
                    for cls in model.classes_
                ],
                "Probability (%)": probabilities * 100
            }).sort_values(
                "Probability (%)",
                ascending=False
            )

            st.subheader("Prediction Probabilities")

            st.dataframe(
                probability_df.style.format({
                    "Probability (%)": "{:.2f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

        # Show submitted values
        with st.expander("View your input values"):
            st.dataframe(
                input_df,
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Check your feature order and ensure the "
            "scaler was fitted on the same input columns."
        )

st.divider()

st.caption(
    "Built with Python, Streamlit and Logistic Regression"
)
