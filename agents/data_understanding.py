import pandas as pd

def data_understanding(df):
    if df.empty:
        return "Error: Uploaded CSV has no data or columns."

    summary = {
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "basic_stats": df.describe(include='all').to_dict()
    }

    return summary
