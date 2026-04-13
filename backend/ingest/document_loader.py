import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document

import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from datetime import datetime

class DocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_all(self) -> List[Document]:
        """Loads all supported files from the data directory."""
        documents = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = Path(root) / file
                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() == '.pdf':
                    documents.extend(self._load_pdf(file_path))
                elif file_path.suffix.lower() in ['.html', '.htm']:
                    documents.extend(self._load_html(file_path))
                elif file_path.suffix.lower() == '.txt':
                    documents.extend(self._load_txt(file_path))
                else:
                    print(f"Skipping unsupported file type: {file_path.name}")
        return documents

    def _get_base_metadata(self, file_path: Path) -> dict:
        """Extract basic metadata including source filename and date."""
        stat_info = os.stat(file_path)
        create_time = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
        return {
            "source": file_path.name,
            "filepath": str(file_path.resolve()),
            "date": create_time
        }

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load a PDF file and extract text and metadata per page."""
        documents = []
        base_metadata = self._get_base_metadata(file_path)
        
        try:
            doc = fitz.open(file_path)
            title = doc.metadata.get("title", "")
            if not title:
                title = file_path.stem
                
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                metadata = base_metadata.copy()
                metadata.update({
                    "title": title,
                    "page": page_num + 1,
                    "document_type": "pdf"
                })
                
                if text.strip():
                    documents.append(Document(page_content=text, metadata=metadata))
            doc.close()
            print(f"Loaded {len(documents)} pages from PDF: {file_path.name}")
        except Exception as e:
            print(f"Error loading PDF {file_path.name}: {e}")
            
        return documents

    def _load_html(self, file_path: Path) -> List[Document]:
        """Load HTML file, extract text and basic structure metadata."""
        documents = []
        base_metadata = self._get_base_metadata(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
                
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else file_path.stem
            
            # Using get_text to extract visible text
            text = soup.get_text(separator="\n", strip=True)
            
            metadata = base_metadata.copy()
            metadata.update({
                "title": title,
                "document_type": "html"
            })
            
            if text.strip():
                documents.append(Document(page_content=text, metadata=metadata))
            print(f"Loaded HTML: {file_path.name}")
        except Exception as e:
            print(f"Error loading HTML {file_path.name}: {e}")
            
        return documents

    def _load_txt(self, file_path: Path) -> List[Document]:
        """Load plain text files."""
        documents = []
        base_metadata = self._get_base_metadata(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
                
            metadata = base_metadata.copy()
            metadata.update({
                "title": file_path.stem,
                "document_type": "txt"
            })
            
            if text.strip():
                documents.append(Document(page_content=text, metadata=metadata))
            print(f"Loaded TXT: {file_path.name}")
        except Exception as e:
            print(f"Error loading TXT {file_path.name}: {e}")
            
        return documents
