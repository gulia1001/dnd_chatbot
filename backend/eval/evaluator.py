import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.vector_store import VectorStoreManager
from generation.llm_chain import GenerationChain

def load_eval_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # Expected columns: question, ground_truth
    return df

def run_evaluation_experiment(chunk_size: int, top_k: int, is_baseline=True):
    """
    Component 5 / Experiment Log script.
    Runs the pipeline over the eval dataset and computes RAGAS metrics.
    """
    print(f"--- Running Experiment | Chunk Size: {chunk_size} | Top-K: {top_k} ---")
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    VDB_NAME = "vdb_storage"
    EVAL_CSV = os.path.join(ROOT_DIR, "eval_dataset.csv")
    LOG_CSV = os.path.join(ROOT_DIR, "experiment_log.csv")
    
    vector_store = VectorStoreManager(persist_dir=VDB_NAME, model_name="all-MiniLM-L6-v2")
    llm_chain = GenerationChain(model_name="gpt-4o-mini", temperature=0.0)
    
    df = load_eval_data(EVAL_CSV)
    
    questions = df["question"].tolist()
    # Note: RAGAS expects lists of strings for ground truths per question in newer versions, 
    # but strings in older ones. We'll wrap in lists if needed.
    ground_truths = [[str(gt)] for gt in df["ground_truth"].tolist()] 
    
    answers = []
    contexts = []
    
    print("Generating responses for eval dataset...")
    for index, q in enumerate(questions):
        # Retrieve context
        docs = vector_store.retrieve(q, top_k=top_k)
        ctx = [d.page_content for d in docs]
        contexts.append(ctx)
        
        # Generate Answer
        ans = llm_chain.generate_answer(q, docs)
        answers.append(ans)
        
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    print("Running RAGAS Evaluation... Ensure OPENAI_API_KEY is exported.")
    try:
        result = evaluate(
            dataset = dataset, 
            metrics=[
                faithfulness,
                answer_relevancy,
            ],
        )
        print("\n=== Experiment Results ===")
        print(result)
        
        # Log to experiment log sheet
        log_entry = f"{'Baseline' if is_baseline else 'Variant'},{chunk_size},{top_k},{result['faithfulness']},{result['answer_relevancy']}\n"
        with open(LOG_CSV, "a") as f:
            f.write(log_entry)
            
    except Exception as e:
        print(f"Failed to run RAGAS evaluation: {e}")

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    LOG_CSV = os.path.join(ROOT_DIR, "experiment_log.csv")
    # Ensure experiment_log.csv has a header
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w") as f:
            f.write("Experiment,ChunkSize,TopK,Faithfulness,AnswerRelevancy\n")
            
    # Run a test experiment
    run_evaluation_experiment(chunk_size=500, top_k=8, is_baseline=True)
