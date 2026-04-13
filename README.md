# D&D RAG Chatbot

This is a production-ready Retrieval-Augmented Generation pipeline built with Python (FastAPI/LangChain) and Node.js (React/Vite). Built specifically against a D&D knowledge base.

## Project Structure
- `backend/`: Python API and RAG logic
  - `ingest/`: Document loaders and chunking logic
  - `retrieval/`: Vector store (Chroma) and embedding model initialization
  - `generation/`: Bounded LLM generation and grounding
  - `eval/`: Evaluate metrics using Ragas
- `frontend/`: Premium React interface with dark-mode formatting
- `data/`: Folder containing raw documents
- `chroma_db/`: Local vector storage generated after ingestion

## Setup

1. Activate your python virtual environment:
```powershell
.\dnd-env\Scripts\Activate.ps1
```

2. Ensure your `.env` file is in the root project folder containing your API key:
```env
OPENAI_API_KEY="sk-your-key-here"
```

## Running the System

### Backend Server
1. Navigate to your backend API folder:
```powershell
cd backend/api
```
2. Start the FastAPI server:
```powershell
python main.py
```
The server will boot on `http://localhost:8000`.

### Frontend Server
1. Open a new terminal and navigate to the frontend folder:
```powershell
cd frontend
```
2. Start Vite development server:
```powershell
npm run dev
```

## Ingesting Documents 
Before you can chat, you must ingest the `data/` folder into your Chroma database. 
Make sure your backend server is running, and then trigger the endpoint:
```powershell
Invoke-WebRequest -Method POST -Uri "http://localhost:8000/ingest"
```
Wait a couple of minutes and watch your backend console as it loads tools, chunks paragraphs, and vectorizes the models via `sentence-transformers/all-MiniLM-L6-v2`.

## Experimentation Log
See `experiment_log.csv` and `eval_dataset.csv` for details on tuning chunk sizes. You can run the testing harness via:
```powershell
cd backend/eval
python evaluator.py
```
