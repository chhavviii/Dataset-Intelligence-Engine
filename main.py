from agents.data_understanding import data_understanding
from agents.insight_generation import generate_insights
from agents.explanation_agent import explain_insights

CSV_FILE = "sample_data.csv"

# 1️⃣ Data Understanding
summary = data_understanding(CSV_FILE)
print("=== Dataset Summary ===")
print(summary)

# 2️⃣ Insight Generation
insights = generate_insights(summary, dataset_name=CSV_FILE)
print("\n=== AI Insights ===")
print(insights)

# 3️⃣ Explanation Agent
plain_english = explain_insights(insights)
print("\n=== Plain-English Explanation ===")
print(plain_english)
