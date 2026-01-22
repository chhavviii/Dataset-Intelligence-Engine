import os
from dotenv import load_dotenv
from openai import OpenAI
from agents.mock_ai import mock_explain_insights

load_dotenv()

def explain_insights(insights):
    mode = os.getenv("AI_MODE", "auto")
    api_key = os.getenv("OPENAI_API_KEY")

    if mode == "mock" or not api_key:
        return mock_explain_insights(insights)

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": insights}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content

    except Exception:
        return mock_explain_insights(insights)
