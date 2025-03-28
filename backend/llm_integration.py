# backend/llm_integration.py
import google.generativeai as genai
import json
import pandas as pd
from config import Config
import logging
import traceback

logger = logging.getLogger(__name__)

try:
    genai.configure(api_key=Config.LLM_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    # Remove explicit reference to EmbeddingModel

    logger.info("Gemini AI configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini AI: {e}", exc_info=True)
    raise  # Re-raise to prevent the app from starting if LLM fails

CHART_PROMPT = """
Given a user query '{query}' and available dataset columns {columns}, suggest a chart type (line, bar, pie, scatter, heatmap) and the X and Y axis labels. Respond in JSON format with the keys: "chart_type", "x", and "y".
"""

TREND_PROMPT = """
Given a user query '{query}' and a dataset with columns {columns}, analyze the data and provide insights or trends.
"""

def interpret_query(query: str, columns: list) -> dict:
    """Interprets a user query and suggests a chart configuration using Gemini."""
    if not model:
        logger.warning("Gemini AI model not initialized. Returning default config.")
        return {"chart_type": "scatter", "x": columns[0], "y": columns[1]} if len(columns) >= 2 else {"chart_type": "scatter", "x": columns[0], "y": None}
    try:
        full_prompt = CHART_PROMPT.format(query=query, columns=columns)
        response = model.generate_content(full_prompt, safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ])

        response_text = response.text.strip()

        try:
            chart_config = json.loads(response_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON. Returning default config.")
            return {"chart_type": "scatter", "x": columns[0], "y": columns[1]} if len(columns) >= 2 else {"chart_type": "scatter", "x": columns[0], "y": None}

        if not isinstance(chart_config, dict) or not all(key in chart_config for key in ["chart_type", "x", "y"]):
            logger.warning("Incomplete chart config from LLM. Returning default.")
            return {"chart_type": "scatter", "x": columns[0], "y": columns[1]} if len(columns) >= 2 else {"chart_type": "scatter", "x": columns[0], "y": None}


        if chart_config["x"] not in columns or chart_config["y"] not in columns:
            logger.warning("Suggested axes not in dataset columns. Returning default.")
            return {"chart_type": "scatter", "x": columns[0], "y": columns[1]} if len(columns) >= 2 else {"chart_type": "scatter", "x": columns[0], "y": None}

        logger.info(f"Gemini AI interpreted query '{query}' as: {chart_config}")
        return chart_config


    except Exception as e:
        logger.error(f"LLM error: {e}", exc_info=True) # include traceback
        return {"chart_type": "scatter", "x": columns[0], "y": columns[1]} if len(columns) >= 2 else {"chart_type": "scatter", "x": columns[0], "y": None}


def analyze_data(query: str, df: pd.DataFrame) -> str:
    """Analyzes data using the LLM, providing robust error handling.

    Returns a default message on failure.
    """
    if not model:
        logger.warning("Gemini AI model not initialized. Returning default message.")
        return "Unable to analyze trends at this time."
    try:
        full_prompt = TREND_PROMPT.format(
            query=query, columns=df.columns.tolist()
        )
        response = model.generate_content(full_prompt, safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ])
        return response.text.strip()
    except Exception as e:
        logger.error(f"Trend analysis error: {e}", exc_info=True) # log traceback
        return "Unable to analyze trends at this time."


def generate_embedding(text: str) -> list:
    """Generates an embedding vector for the text using Gemini.
    Handles exceptions and returns None upon failure, rather than raising.
    """
    if not model:
        logger.warning("Gemini AI model not initialized. Returning None.")
        return None

    try:
        response = genai.embed_content(
            model=Config.EMBEDDING_MODEL_NAME,  # Use the configured model name
            content=text,
            task_type="retrieval_query",  # Or appropriate task type
            title="Embedding for Chart Data",
        )

        return response["embedding"] # Access the embedding directly
    except Exception as e:
        logger.error(f"Failed to generate embedding for text: {text[:100]}...: {e}", exc_info=True)
        return None # handle embedding failures gracefully