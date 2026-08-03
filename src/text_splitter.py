"""
text_splitter.py

Splits LangChain Documents into smaller chunks while preserving metadata.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch import chunk

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextSplitter:

    def __init__(self):
        """
        Initialize Recursive Character Text Splitter.
        """

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
        )

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents:
                List of LangChain Document objects.

        Returns:
            List of chunked Documents.
        """

        chunks = self.splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["chunk_length"] = len(chunk.page_content)

        return chunks