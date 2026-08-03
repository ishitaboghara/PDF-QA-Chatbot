"""
vector_db.py

Creates, saves, loads, and searches a FAISS vector database.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import VECTORSTORE_DIR
from src.embeddings import EmbeddingModel


class VectorDatabase:

    def __init__(self):

        self.embedding_model = EmbeddingModel().get_model()

        self.db_path = VECTORSTORE_DIR

        self.vectorstore = None

    def create_vectorstore(
        self,
        documents: List[Document]
    ) -> FAISS:
        """
        Create FAISS index from documents.
        """

        self.vectorstore = FAISS.from_documents(
            documents,
            self.embedding_model
        )

        return self.vectorstore

    def save(self):
        """
        Save FAISS index locally.
        """

        if self.vectorstore is None:
            raise ValueError("Vector store has not been created.")

        self.vectorstore.save_local(str(self.db_path))

    def load(self):
        """
        Load saved FAISS index.
        """

        self.vectorstore = FAISS.load_local(
            str(self.db_path),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        return self.vectorstore

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ):
        """
        Perform semantic similarity search.
        """

        if self.vectorstore is None:
            raise ValueError("Vector store is not loaded.")

        return self.vectorstore.similarity_search(
            query,
            k=k
        )