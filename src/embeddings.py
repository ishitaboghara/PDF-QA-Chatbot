"""
embeddings.py

Creates and manages the Hugging Face embedding model.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, EMBEDDING_DEVICE


class EmbeddingModel:
    """
    Wrapper around Hugging Face sentence-transformer embeddings.
    """

    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={
                "device": EMBEDDING_DEVICE
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    def get_model(self):
        """
        Returns the initialized embedding model.
        """
        return self.model