# Dataset-Intelligence-Engine
An intelligent, agent-based **AI Data Analyst** application that automatically understands uploaded CSV datasets, recommends **dataset-specific EDA steps**, identifies the **correct machine learning problem type**, and suggests **appropriate ML models** — all through an interactive **Streamlit UI**.

Built using **Python, Streamlit, and a CrewAI-style multi-agent architecture**.

---

## 🚀 Features

### 📂 CSV Upload & Profiling
- Upload any tabular CSV dataset
- Automatically detects:
  - Numerical columns
  - Categorical columns
  - Datetime columns (robust multi-layer detection)
  - Identifier-like columns

---

### 🤖 Intelligent ML Strategy Recommendation
- Automatically infers:
  - Target variable (with human confirmation)
  - ML problem type:
    - Classification
    - Regression
    - Time Series
    - Unsupervised Learning
- Recommends **dataset-aware ML models**
  - Binary vs multi-class classification
  - High-dimensional vs low-dimensional regression
  - Time-series specific models

---

### 📊 Dataset-Specific EDA Recommendations (Non-Generic)
Unlike rule-based EDA tools, this project:
- Analyzes **actual data statistics**
- Detects:
  - Class imbalance
  - Skewed numeric features
  - High-cardinality categorical columns
  - Duplicate rows
  - High missing-value columns
- Generates **EDA steps that change per dataset**

---

### 🧠 Human-in-the-Loop Design
- Auto-detected target variable can be **confirmed or corrected** by the user
- ML strategy updates accordingly
- Ensures transparency and correctness

---

### 💡 AI Insights & Explanation
- Generates high-level insights about the dataset
- Explains recommendations in **plain English**
- Includes a **mock mode** for demo purposes (no API credits required)

---
## 🏗️ Project Structure
AI-DATA-ANALYST/
│
├── app.py # Streamlit application
├── requirements.txt
├── agents/
│ ├── data_understanding.py # Dataset summary agent
│ ├── ml_strategy_agent.py # ML + EDA recommendation agent
│ ├── insight_generation.py # AI insight agent
│ └── explanation_agent.py # Plain-English explanation agent
│
└── .env (optional)


---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit**
- **Pandas / NumPy**
- **CrewAI-style agent design**
- **OpenAI API (optional – mock mode supported)**

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/chhavviii/ai-data-analyst-crewai.git
cd ai-data-analyst-crewai
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ (Optional) Add OpenAI API key
```bash 
OPENAI_API_KEY=your_api_key_here
```
### 4️⃣ Run the Streamlit app
```bash
streamlit run app.py
```



