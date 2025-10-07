import os
from openai import OpenAI
from typing import List, Tuple

class QAEngine:
    #Initialize API key
    def __init__(self, llm_model: str = "gpt-3.5-turbo",api_key: str = None):
        self.llm_model = llm_model
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")

    def build_prompt(self, question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
        """Build a single prompt string for the LLM from retrieved chunks."""
        context_text = ""
        for chunk_text,score in retrieved_chunks:
            context_text += f"{chunk_text}\n---\n"
        #Combine all context chunks with the user question into a single LLM prompt    
        prompt = f"You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context_text}\nQuestion: {question}\nAnswer:"
        return prompt

    def generate_answer(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
        """Generate an answer from the LLM using the 1.x API."""
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}], ## User message for the LLM
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
