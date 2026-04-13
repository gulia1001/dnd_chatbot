import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

class GenerationChain:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Initializes the LLM connection to OpenAI.
        Expects OPENAI_API_KEY environment variable.
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        Component 5: Generation & Grounding.
        Generates an answer ensuring adherence to grounding and citation rules.
        """
        if not context_docs:
            return "I cannot find this in the provided documents."

        # Process context blocks with metadata formatting
        context_text = ""
        for i, doc in enumerate(context_docs):
            src = doc.metadata.get('source', f'Doc_{i}')
            page = doc.metadata.get('page', '')
            title = doc.metadata.get('title', '')
            
            ref = f"[{src}]" if not page else f"[{src}, Page {page}]"
            if title and title != src:
                ref += f" (Title: {title})"
                
            context_text += f"\n--- Source: {ref} ---\n{doc.page_content}\n"

        system_prompt = (
            "You are an expert Q&A assistant for a tabletop RPG (Dungeons and Dragons). "
            "Your task is to answer the user's question based strictly on the provided context documents.\n\n"
            "Rules for answering:\n"
            "1. Answer ONLY using the provided context. If the context does not fully cover the entire question, answer with whatever portion of the context is related to the query.\n"
            "2. You MUST cite the source document by name for every factual claim you make, explicitly using the source brackets provided (e.g., '[Player\\'s Handbook.pdf, Page 12]').\n"
            "3. If the retrieved documents literally do not contain any information remotely related to the user's query, state 'I cannot find this in the provided documents.' AND THEN provide a brief 1-sentence summary of what the retrieved text actually says so the user knows what was extracted by the parser.\n\n"
            "Context Documents:\n"
            f"{context_text}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Query: {query}")
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return "Error during generation process."
