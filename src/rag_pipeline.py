"""
rag_pipeline.py

Production-ready RAG pipeline.
"""

from src.vector_db import VectorDatabase
from src.llm import LocalLLM
from src.prompt import RAG_PROMPT
from src.config import TOP_K_RESULTS


class RAGPipeline:

    def __init__(self):

        self.vector_db = VectorDatabase()
        self.vector_db.load()

        self.llm = LocalLLM().get_llm()

    def ask(self, question: str):

        docs = self.vector_db.similarity_search(
            question,
            k=TOP_K_RESULTS
        )

        # Keep prompt within FLAN-T5 limits
        context = ""

        for doc in docs:

            remaining = 1200 - len(context)

            if remaining <= 0:
                break

            context += doc.page_content[:remaining]
            context += "\n\n"

        prompt = RAG_PROMPT.format(
            context=context,
            question=question
        )

        answer = self.llm.invoke(prompt)

        sources = []

        seen = set()

        for doc in docs:

            key = (
                doc.metadata["source"],
                doc.metadata["page"]
            )

            if key not in seen:

                seen.add(key)

                sources.append({
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"]
                })

        return answer, sources