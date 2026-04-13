import os
import sys
import traceback
from dotenv import load_dotenv

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT_DIR, "backend"))
load_dotenv()

from backend.generation.llm_chain import GenerationChain
from backend.retrieval.vector_store import VectorStoreManager

def main():
    print("================================")
    print("  D&D CHATBOT DIAGNOSTIC TEST   ")
    print("================================")
    
    print("\n[1/3] Testing OpenAI Connection...")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY is not defined in .env!")
        else:
            print(f"API Key found (starts with: {api_key[:7]}...)")
            
        llm = GenerationChain()
        # Test basic invocation
        from langchain_core.messages import HumanMessage
        print("Sending ping to OpenAI...")
        resp = llm.llm.invoke([HumanMessage(content="Reply 'Ping successful' if you can read this.")])
        print(f"OpenAI Response: {resp.content}")
        print("=> OpenAI Connection works!")
    except Exception as e:
        print(f"WARNING! OpenAI Connection failed: {e}")
        traceback.print_exc()

    print("\n[2/3] Testing Vector Store Loading...")
    try:
        VDB_NAME = "vdb_storage"
        print(f"Loading Vector Store from relative path: {VDB_NAME}...")
        vm = VectorStoreManager(persist_dir=VDB_NAME)
        print("=> Vector Store loaded!")
    except Exception as e:
        print(f"WARNING! Vector Store threw an error on load: {e}")
        traceback.print_exc()
        return

    print("\n[3/3] Testing Hybrid Search retrieval...")
    try:
        query = "give 5 names of monsters whose name starting with letter A"
        docs = vm.retrieve(query, 8)
        print(f"Retrieved {len(docs)} documents.")
        print("====== RETRIEVED SNIPPETS ======")
        for i, d in enumerate(docs):
            print(f"{i+1}. [{d.metadata.get('source')}] | {repr(d.page_content[:150])}")
        print("=> Retrieval works!")
    except Exception as e:
        print(f"WARNING! Retrieval threw an error: {e}")
        traceback.print_exc()
        
    print("\n[4/4] Testing End-To-End generation...")
    try:
        answer = llm.generate_answer(query, docs)
        print(f"Answer: {answer[:300]}...")
        print("=> Generation works!")
    except Exception as e:
        print(f"WARNING! Endpoint Generate threw an error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
