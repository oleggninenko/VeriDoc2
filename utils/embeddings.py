"""
Embedding utilities for text processing.
Consolidated from duplicate implementations across the codebase.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def embed_texts_batch(client, model: str, inputs: list) -> np.ndarray:
    """
    Embed a batch of texts using the specified model.
    
    Args:
        client: OpenAI client instance
        model (str): Model name for embeddings
        inputs (list): List of text strings to embed
        
    Returns:
        np.ndarray: Array of embeddings
    """
    try:
        # Process in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            response = client.embeddings.create(
                model=model,
                input=batch
            )
            batch_embeddings = [embedding.embedding for embedding in response.data]
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(inputs) + batch_size - 1)//batch_size}")
        
        return np.array(all_embeddings)
        
    except Exception as e:
        logger.error(f"Error in batch embedding: {str(e)}")
        raise


def embed_single_text(client, model: str, text: str) -> np.ndarray:
    """
    Embed a single text using the specified model.
    
    Args:
        client: OpenAI client instance
        model (str): Model name for embeddings
        text (str): Text string to embed
        
    Returns:
        np.ndarray: Single embedding vector
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return np.array(response.data[0].embedding)
        
    except Exception as e:
        logger.error(f"Error in single text embedding: {str(e)}")
        raise

