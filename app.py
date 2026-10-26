from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Kyphosis Prediction | ML Demo", page_icon="🧠", layout="wide")
st.title("Predicting Presence and Absence of Kyphosis")
st.caption("AI & Data Science II | Decision Tree and Random Forest")
st.warning("Educational demonstration only. This model is not validated for medical diagnosis or treatment decisions.")

@st.cache_resource
def load_models():
    return {
        "Decision Tree": joblib.load(BASE / "models" / "decision_tree.joblib"),
        "Random Forest": joblib.load(BASE / "models" / "random_forest.joblib"),
    }

models = load_models()
df = pd.read_csv(BASE / "data" / "kyphosis.csv")
with st.sidebar:
    st.header("Patient inputs")
    model_name = st.selectbox("Choose classifier", list(models))
    age = st.number_input("Age (months)", min_value=1, max_value=240, value=71, step=1)
    number = st.number_input("Number of vertebrae involved", min_value=1, max_value=20, value=3, step=1)
    start = st.number_input("Start (topmost vertebra)", min_value=1, max_value=30, value=5, step=1)
    predict = st.button("Predict", type="primary")

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Dataset overview")
    st.write(f"**{len(df)}** patient records | **{df.shape[1]}** columns")
    st.bar_chart(df["Kyphosis"].value_counts())
    st.dataframe(df.head(10), hide_index=True)
with col2:
    st.subheader("Model information")
    st.write(f"Selected: **{model_name}**")
    st.write("Features: Age, Number, Start")
    st.write("Target: kyphosis absent or present after surgery")
    st.write("Train/test split: 70/30, stratified, random_state=42")
    st.caption("The submitted report used a different or unspecified training configuration and reported 84% accuracy. New evaluation results are in outputs/metrics.json.")

if predict:
    sample = pd.DataFrame([[age, number, start]], columns=["Age", "Number", "Start"])
    model = models[model_name]
    predicted = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0, 1])
    st.divider()
    st.subheader("Prediction result")
    st.metric("Predicted class", "PRESENT" if predicted else "ABSENT")
    st.write(f"Model estimated probability for 'present': **{probability:.1%}**")
    st.info("This is a model output from a small historical dataset, not a clinical risk estimate.")
