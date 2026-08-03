"""
prompt.py

Prompt template used by the RAG pipeline.
"""

from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate.from_template(
"""
You are an intelligent AI assistant that answers questions ONLY using the provided context.

Instructions:

1. Read the context carefully.
2. Answer ONLY from the provided context.
3. Do NOT make up information.
4. If the answer is not present in the context, reply exactly:

"I couldn't find the answer in the uploaded PDF."

5. Keep the answer concise but complete.
6. If possible, explain in simple language.

-------------------------
Context:

{context}

-------------------------

Question:

{question}

-------------------------

Answer:
"""
)