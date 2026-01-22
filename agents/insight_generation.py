import os
from dotenv import load_dotenv
from openai import OpenAI
from agents.mock_ai import mock_generate_insights

load_dotenv()

def generate_insights(summary, dataset_name="dataset"):
    mode = os.getenv("AI_MODE", "auto")
    api_key = os.getenv("OPENAI_API_KEY")

    if mode == "mock" or not api_key:
        return mock_generate_insights(summary, dataset_name)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": str(summary)}
            ],
            max_tokens=400
        )
        return response.choices[0].message.content

    except Exception:
        return mock_generate_insights(summary, dataset_name)
