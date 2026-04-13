import os
import sys
import shutil

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT_DIR, "backend"))

from backend.ingest.document_loader import DocumentLoader
from backend.ingest.chunker import DocumentChunker
from backend.retrieval.vector_store import VectorStoreManager

DATA_DIR = os.path.join(ROOT_DIR, "data")
VDB_NAME = "vdb_storage"

def run():
    print("====================================")
    print("  D&D RAG CHATBOT INGESTION SCRIPT  ")
    print("====================================")
    
    # Clean the old corrupted database
    if os.path.exists(VDB_NAME):
        print(f"Cleaning out old database at {VDB_NAME}...")
        try:
            shutil.rmtree(VDB_NAME)
            print("Old database deleted successfully.")
        except Exception as e:
            print(f"FAILED TO DELETE DATABASE. Error: {e}")
            return
    
    # Also clean the old folder just in case
    old_dir = os.path.join(ROOT_DIR, "chroma_db")
    if os.path.exists(old_dir):
        try: shutil.rmtree(old_dir)
        except: pass
    
    print("\n[Step 1/3] Loading PDFs and HTMLs. This might take a minute...")
    loader = DocumentLoader(DATA_DIR)
    docs = loader.load_all()
    
    print(f"\n[Step 2/3] Chunking {len(docs)} documents...")
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100, strategy="recursive")
    chunks = chunker.chunk_documents(docs)
    
    print(f"\n[Step 3/3] Vectorizing {len(chunks)} chunks into Hybrid Store (FAISS + BM25).")
    print("WARNING: DO NOT CLOSE THIS TERMINAL. IT CAN TAKE 2-5 MINUTES.")
    vector_store_manager = VectorStoreManager(persist_dir=VDB_NAME, model_name="all-MiniLM-L6-v2")
    vector_store_manager.add_documents(chunks)
    
    print("\n====================================")
    print("  INGESTION COMPLETED SUCCESSFULLY! ")
    print("====================================")

if __name__ == "__main__":
    run()
