"""
llm.py

Initializes the local Hugging Face FLAN-T5 model.
"""

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

from src.config import (
    LLM_MODEL,
    MAX_NEW_TOKENS,
    LLM_DEVICE,
)


class LocalLLM:

    def __init__(self):

        generator = pipeline(
            task="text2text-generation",
            model=LLM_MODEL,
            tokenizer=LLM_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            device=LLM_DEVICE,
        )

        self.llm = HuggingFacePipeline(
            pipeline=generator
        )

    def get_llm(self):
        return self.llm