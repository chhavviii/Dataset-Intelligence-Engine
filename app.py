import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from agents.data_understanding import data_understanding
from agents.insight_generation import generate_insights
from agents.explanation_agent import explain_insights
from agents.ml_strategy_agent import recommend_ml_strategy

# =========================
# Environment setup
# =========================
load_dotenv()

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("🧠 AI Data Analyst (CrewAI)")

# =========================
# Sidebar controls
# =========================
mode = st.sidebar.selectbox(
    "AI Mode",
    ["auto", "mock", "live"]
)
os.environ["AI_MODE"] = mode

# =========================
# File upload
# =========================
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # =========================
    # 1️⃣ Data Understanding
    # =========================
    st.subheader("📊 Dataset Summary")
    summary = data_understanding(df)
    st.write(summary)

    # =========================
    # 2️⃣ Initial ML Strategy
    # =========================
    st.subheader("🤖 ML Strategy Recommendation (Auto-detected)")
    ml_strategy = recommend_ml_strategy(df)
    st.json(ml_strategy)

    # =========================
    # 3️⃣ Target confirmation (IMPORTANT)
    # =========================
    st.subheader("🎯 Confirm Target Variable")

    guessed_target = ml_strategy.get("target_variable_guess")

    if guessed_target in df.columns:
        default_index = df.columns.get_loc(guessed_target)
    else:
        default_index = 0

    selected_target = st.selectbox(
        "Select / Confirm the target column",
        df.columns,
        index=default_index
    )

    st.info(f"Selected target variable: **{selected_target}**")

    # =========================
    # 4️⃣ Re-run ML strategy using confirmed target
    # =========================
    st.subheader("🔁 ML Strategy (After Target Confirmation)")

    df_with_target = df.copy()
    ml_strategy_final = recommend_ml_strategy(df_with_target)

    # Override guessed target with user-selected target
    ml_strategy_final["target_variable_confirmed"] = selected_target

    st.json(ml_strategy_final)

    # =========================
    # 5️⃣ AI Insights
    # =========================
    st.subheader("💡 AI Insights")

    try:
        insights = generate_insights(
            summary,
            dataset_name=uploaded_file.name
        )
        st.write(insights)
    except Exception as e:
        st.error(f"AI Insight Error: {e}")
        insights = "MOCK MODE: Insights could not be generated."
        st.write(insights)

    # =========================
    # 6️⃣ Plain-English Explanation
    # =========================
    st.subheader("📝 Plain-English Explanation")

    try:
        explanation = explain_insights(insights)
        st.write(explanation)
    except Exception as e:
        st.error(f"Explanation Error: {e}")
        st.write("MOCK MODE: Explanation unavailable.")

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
