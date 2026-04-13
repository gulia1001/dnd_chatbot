import os
import pickle
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from flashrank import Ranker, RerankRequest

class VectorStoreManager:
    def __init__(self, persist_dir: str = "vdb_storage", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Hybrid Search pipeline with FlashRank Reranking.
        Combines Semantic (FAISS) and Keyword (BM25), then re-scores with a Cross-Encoder.
        """
        print(f"Loading embedding model: {model_name}...")
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.bm25_retriever = None
        self.ranker = None
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
                self.bm25_retriever.k = 8
                
            # 3. Initialize FlashRank (Reranker)
            print("Initializing local Reranker (FlashRank)...")
            self.ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="rerank_cache")
        else:
            print("No existing index found. Hybrid Search + Reranking will be available after ingestion.")

    def add_documents(self, documents: List[Document]):
        """Embeds and indexes chunks, then caches them for BM25."""
        if not documents:
            return
            
        print(f"Indexing {len(documents)} chunks...")
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
            
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
            
        self.vector_store.save_local(self.persist_dir)
        
        with open(self.chunks_path, "wb") as f:
            pickle.dump(documents, f)
            
        print(f"Success! Hybrid DB saved in: {os.path.abspath(self.persist_dir)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Advanced Retrieval Strategy:
        1. Retrieve candidates (top_k * 3) from FAISS and BM25.
        2. Deduplicate.
        3. Rerank candidates using FlashRank.
        4. Return top reranked results.
        """
        candidates = []
        seen_texts = set()

        # Phase 1: Candidate Gathering
        if self.vector_store:
            semantic_docs = self.vector_store.similarity_search(query, k=top_k * 2)
            for d in semantic_docs:
                if d.page_content not in seen_texts:
                    candidates.append(d)
                    seen_texts.add(d.page_content)

        if self.bm25_retriever:
            keyword_docs = self.bm25_retriever.get_relevant_documents(query)
            for d in keyword_docs:
                if d.page_content not in seen_texts:
                    candidates.append(d)
                    seen_texts.add(d.page_content)

        # Phase 2: Reranking
        if self.ranker and candidates:
            # Prepare data for FlashRank
            passages = [
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(candidates)
            ]
            
            rerank_request = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(rerank_request)
            
            # Map back to LangChain Documents
            reranked_docs = []
            for res in results[:top_k]:
                reranked_docs.append(Document(
                    page_content=res["text"],
                    metadata=res["meta"]
                ))
            return reranked_docs

        return candidates[:top_k]
