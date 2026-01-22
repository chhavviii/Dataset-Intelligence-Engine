def mock_generate_insights(summary, dataset_name="dataset"):
    return f"""
1. The dataset '{dataset_name}' contains {len(summary['columns'])} columns.
2. Some columns have missing values, which may affect analysis accuracy.
3. Numerical columns show variation, indicating diverse data patterns.
4. Data cleaning and feature selection would improve model readiness.
5. The dataset is suitable for exploratory and predictive analysis.
"""

def mock_explain_insights(insights):
    return """
This dataset has multiple columns with different types of data.
Some values are missing, so cleaning is needed before analysis.
Overall, the data shows useful patterns and can support decision-making.
"""
