# backend/database.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import pandas as pd
import numpy as np
from config import Config
import logging

logger = logging.getLogger(__name__)

client = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
COLLECTION_NAME = "charts"
VECTOR_SIZE = 768  # Adjust as needed
PAYLOAD_LIMIT = 100  # Max results to retrieve

# Ensure collection exists on startup
def create_collection(collection_name: str = COLLECTION_NAME, vector_size: int = VECTOR_SIZE):
    """Creates a Qdrant collection if it doesn't exist."""
    try:
        client.get_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logger.info(f"Creating collection '{collection_name}' with vector size {vector_size}...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name}': {e}")
        raise

create_collection()

def save_embeddings(df: pd.DataFrame, embeddings: list, collection_name: str = COLLECTION_NAME, filename: str = None):
    """Saves embeddings to Qdrant.  Includes filename in payload

    Args:
        df: DataFrame containing the data.
        embeddings: List of embeddings for each row in the DataFrame.
        collection_name: Name of the Qdrant collection.
        filename: name of the file the df is loaded from, if applicable
    """
    try:
        points = []
        for i in range(len(df)):
            payload = df.iloc[i].to_dict()  # Convert row to dictionary

            if filename:
                payload["filename"] = filename

            vector = embeddings[i]
            points.append(PointStruct(id=hash((i, filename)), vector=vector, payload=payload))

        client.upsert(collection_name=collection_name, wait=True, points=points)
        logger.info(f"Saved {len(embeddings)} embeddings to collection '{collection_name}'.")
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}", exc_info=True)
        raise

def retrieve_data(query_vector: list, query: str, limit: int = 5, collection_name: str = COLLECTION_NAME):
    """Retrieves data from Qdrant based on a query vector.

    Args:
        query_vector: The vector to search for similar data.
        query: the search query
        limit: The maximum number of results to return.
        collection_name: Name of the Qdrant collection.

    Returns:
        A list of search results.
    """
    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )

        logger.info(f"Retrieved {len(search_result)} results from collection '{collection_name}' for query '{query}'.")
        return search_result

    except Exception as e:
        logger.error(f"Error retrieving data: {e}", exc_info=True)
        return []