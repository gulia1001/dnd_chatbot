import os
import pickle
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

class VectorStoreManager:
    def __init__(self, persist_dir: str = "vdb_storage", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Hybrid Search pipeline.
        Manually combines FAISS (Semantic) and BM25 (Keyword) to avoid dependency errors.
        """
        print(f"Loading embedding model: {model_name}...")
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.bm25_retriever = None
        self.chunks_path = os.path.join(self.persist_dir, "chunks.pkl")
        
        # 1. Load FAISS (Semantic)
        if os.path.exists(os.path.join(self.persist_dir, "index.faiss")):
            print(f"Loading FAISS index from relative path: {self.persist_dir}...")
            self.vector_store = FAISS.load_local(
                folder_path=self.persist_dir, 
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # 2. Load BM25 (Keyword) if chunks are cached
            if os.path.exists(self.chunks_path):
                print("Building Keyword Retriever (BM25)...")
                with open(self.chunks_path, "rb") as f:
                    chunks = pickle.load(f)
                
                self.bm25_retriever = BM25Retriever.from_documents(chunks)
                self.bm25_retriever.k = 5
        else:
            print("No existing index found. Hybrid Search will be available after ingestion.")

    def add_documents(self, documents: List[Document]):
        """Embeds and indexes chunks, then caches them for BM25."""
        if not documents:
            return
            
        print(f"Indexing {len(documents)} chunks...")
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
            
        # Ensure the directory exists
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
            
        # Save FAISS
        self.vector_store.save_local(self.persist_dir)
        
        # Save Raw Chunks for BM25 persistence
        with open(self.chunks_path, "wb") as f:
            pickle.dump(documents, f)
            
        print(f"Success! Hybrid DB saved in: {os.path.abspath(self.persist_dir)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Hybrid Retrieval Logic:
        1. Get Top-K from FAISS (Semantic)
        2. Get Top-K from BM25 (Keyword)
        3. Merge and deduplicate.
        """
        combined_docs = []
        seen_texts = set()

        # 1. Semantic Hits
        if self.vector_store:
            semantic_docs = self.vector_store.similarity_search(query, k=top_k)
            for d in semantic_docs:
                if d.page_content not in seen_texts:
                    combined_docs.append(d)
                    seen_texts.add(d.page_content)

        # 2. Keyword Hits
        if self.bm25_retriever:
            # For list queries, keyword search is often more reliable
            keyword_docs = self.bm25_retriever.get_relevant_documents(query)
            for d in keyword_docs:
                if d.page_content not in seen_texts:
                    combined_docs.append(d)
                    seen_texts.add(d.page_content)

        # Return combined results - Increase pool to top_k * 3 for better list coverage
        return combined_docs[:int(top_k * 3)]
