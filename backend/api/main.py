import os
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Add backend directory to sys.path so modules can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.document_loader import DocumentLoader
from ingest.chunker import DocumentChunker
from retrieval.vector_store import VectorStoreManager
from generation.llm_chain import GenerationChain

app = FastAPI(title="D&D RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
print("Initializing pipeline components...")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
VDB_NAME = "vdb_storage"

vector_store_manager = VectorStoreManager(persist_dir=VDB_NAME, model_name="all-MiniLM-L6-v2")
llm_chain = GenerationChain(model_name="gpt-4o-mini", temperature=0.0)

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Retrieve context (top-8 chunks, results in ~24 combined hits for total coverage)
        retrieved_docs = vector_store_manager.retrieve(request.query, top_k=8)
        
        # Generate Answer based strictly on context
        answer = llm_chain.generate_answer(request.query, retrieved_docs)
        
        # Format sources for UI
        sources = [doc.metadata for doc in retrieved_docs]
        
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_data(background_tasks: BackgroundTasks):
    """
    Triggers the document ingestion manually. Note that paths are relative to where you run this script.
    """
    def run_ingestion():
        print("Starting ingestion process...")
        loader = DocumentLoader(DATA_DIR)
        docs = loader.load_all()
        
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100, strategy="recursive")
        chunks = chunker.chunk_documents(docs)
        
        vector_store_manager.add_documents(chunks)
        print("Ingestion completed successfully.")

    background_tasks.add_task(run_ingestion)
    return {"message": "Ingestion started in the background. Check backend console for progress."}

if __name__ == "__main__":
    import uvicorn
    # run locally on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
