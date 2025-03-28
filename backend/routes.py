# backend/routes.py
import pandas as pd
from io import StringIO
import json
from typing import Dict, Optional
from chart_generator import create_chart
from llm_integration import interpret_query, analyze_data, generate_embedding
from utils import validate_dataset, sanitize_column_name
import logging
import traceback
import plotly
from fastapi.encoders import jsonable_encoder

logger = logging.getLogger(__name__)


def generate_chart(
    query: str, data_content: bytes, chart_type: Optional[str] = None, data_type: str = None, filename: Optional[str] = None
) -> Dict:
    """Generates a chart based on the provided data and query.

    Includes comprehensive error handling and data source identification.

    Args:
        query: The user's query.
        data_content: The data as bytes.
        chart_type: The desired chart type, if specified.
        data_type: String identifier of how the data was provided ('file', 'json', 'manual')
        filename: name of the file if there is a file

    Returns:
        A dictionary containing chart information.

    Raises:
        ValueError: If there are issues with data processing or chart generation.
    """
    try:
        df = None

        # Data loading based on type
        if data_type == "json":
            try:
                df = pd.DataFrame(json.loads(data_content.decode("utf-8")))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON data: {e}") from e

        elif data_type == "file":
            try:
                df = pd.read_csv(StringIO(data_content.decode("utf-8")))
            except Exception as e:
                try:
                   df = pd.DataFrame(json.loads(data_content.decode("utf-8")))
                except Exception as e:
                    raise ValueError(f"Error processing file content as CSV or JSON: {e}") from e

        elif data_type == "manual":
            try:
                df = pd.read_csv(StringIO(data_content.decode("utf-8")))
            except Exception as e:
                raise ValueError(f"Error processing manual CSV data: {e}") from e

        else:
            raise ValueError("Unknown data type.")

        validate_dataset(df) # validation occurs after loading
        # Column name sanitization (before LLM)
        df.columns = [sanitize_column_name(col) for col in df.columns]

        if chart_type:
            chart_config = {"chart_type": chart_type, "x": df.columns[0], "y": df.columns[1]}
        else:
            chart_config = interpret_query(query, df.columns.tolist())

        chart_type = chart_config["chart_type"]
        x_axis = chart_config["x"]
        y_axis = chart_config["y"]

        if x_axis not in df.columns or y_axis not in df.columns:
            raise ValueError(f"Columns '{x_axis}' or '{y_axis}' not found in dataset")

        # Chart generation (delegated to chart_generator.py)
        fig_dict = create_chart(chart_type, df, x_axis, y_axis) #generate the figure directly

        combined_text = f"{query} {df.to_string()} chart type {chart_type} x axis {x_axis} y axis {y_axis}"
        embedding = generate_embedding(combined_text)

        # Save embeddings and metadata to Qdrant
        if embedding:
            try:
                save_embeddings(df, [embedding], filename=filename) # include filename
            except Exception as e:
                logger.warning(f"Failed to save embeddings to Qdrant. {e}", exc_info=True) #non-critical

        logger.info(f"Generated {chart_type} chart for query: {query}")
        # This line fixes the serialization issues
        return {
            "chart_type": chart_type,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "data": df[[x_axis, y_axis]].to_dict(orient="list"), # raw data, useful for debugging
            "plotly_data": json.loads(json.dumps(fig_dict, default=str)),
            "status": "success",
        }

    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}", exc_info=True)
        raise  # Re-raise to be handled by the global exception handler


def analyze_trends(query: str, file_content: bytes) -> Dict:
    """Analyzes trends in the provided CSV data using the LLM.
    Robust error handling ensures graceful failure.
    """
    try:
        df = pd.read_csv(StringIO(file_content.decode("utf-8")))
        validate_dataset(df) # validation occurs after loading
        insights = analyze_data(query, df) # LLM processing

        return {"insights": insights, "status": "success"} #successful result
    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}", exc_info=True)
        raise