from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, strategy: str = "recursive"):
        """
        Initializes the chunker.
        :param chunk_size: Target token size (100 - 512 allowed).
        :param chunk_overlap: Overlap in tokens (10-25%).
        :param strategy: 'recursive' or 'fixed'.
        """
        if chunk_size < 100 or chunk_size > 512:
            print(f"Warning: chunk_size {chunk_size} is outside recommended limits 100-512")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits the provided documents into smaller chunks."""
        print(f"Chunking {len(documents)} documents using '{self.strategy}' strategy (Size: {self.chunk_size}, Overlap: {self.chunk_overlap})...")
        
        if self.strategy == "recursive":
            # Sentence-aware or recursive splitting using token counts
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.strategy == "fixed":
            # Fixed-size chunks with overlap, measured by token counts, using paragraph/newline seps
            splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="\n",
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")
            
        chunks = splitter.split_documents(documents)
        
        # Keep track of indices for each source file
        source_counts = {}
        for chunk in chunks:
            # Metadata was carried over by LangChain splitters automatically
            src = chunk.metadata.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
            chunk.metadata["chunk_index"] = source_counts[src]
            
        print(f"Created {len(chunks)} chunks.")
        return chunks
