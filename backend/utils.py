# backend/utils.py
import pandas as pd

def validate_dataset(df: pd.DataFrame) -> None:
    """Validates the dataset.

    Args:
        df: The Pandas DataFrame to validate.

    Raises:
        ValueError: If the dataset is empty or has fewer than 2 columns, or columns have the same name.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")
    if len(df.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns.")
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Column names must be unique.")


def sanitize_column_name(name: str) -> str:
    """Sanitizes a column name by stripping whitespace and replacing spaces with underscores.
    Lowercases the entire name, to ensure it will work with Gemini and Plotly.
    """
    return name.strip().replace(" ", "_").lower()