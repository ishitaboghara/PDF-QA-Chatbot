"""
pdf_loader.py

Handles loading one or multiple PDF files and converts them
into LangChain Document objects with metadata.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from pypdf import PdfReader


class PDFLoader:
    """
    PDFLoader reads PDF files and returns a list of LangChain
    Document objects.

    Metadata includes:
    - source (PDF filename)
    - page (page number)
    """

    @staticmethod
    def load_pdf(pdf_path: str | Path) -> List[Document]:
        """
        Load a single PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List[Document]
        """

        pdf_path = Path(pdf_path)

        reader = PdfReader(pdf_path)

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            if not text:
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number,
                },
            )

            documents.append(doc)

        return documents

    @staticmethod
    def load_multiple_pdfs(pdf_paths: List[str | Path]) -> List[Document]:
        """
        Load multiple PDFs.

        Args:
            pdf_paths: List of PDF paths

        Returns:
            Combined list of Documents
        """

        all_documents = []

        for pdf in pdf_paths:

            all_documents.extend(
                PDFLoader.load_pdf(pdf)
            )

        return all_documents