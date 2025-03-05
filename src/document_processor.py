"""
Document processor for the AI Education Evidence Library.
Handles ingestion and processing of PDFs and web content.
"""
import os
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.schema.document import Document

import config

class DocumentProcessor:
    """Process documents for the evidence library."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """
        Process a PDF file and return chunked documents.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            List of Document objects with text and metadata.
        """
        try:
            # Extract filename for basic metadata
            filename = os.path.basename(file_path)
            name, _ = os.path.splitext(filename)
            
            # Load and process the PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add basic metadata if not present
            for doc in documents:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_path
                if 'title' not in doc.metadata:
                    doc.metadata['title'] = name
            
            # Split documents into chunks
            chunked_documents = self.text_splitter.split_documents(documents)
            
            return chunked_documents
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return []
    
    def process_web_content(self, url: str) -> List[Document]:
        """
        Process content from a web page.
        
        Args:
            url: URL of the web page.
            
        Returns:
            List of Document objects with text and metadata.
        """
        try:
            # Load and process the web content
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Extract title if possible
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.title.string if soup.title else url
            except:
                title = url
                
            # Add or update metadata
            for doc in documents:
                doc.metadata['source'] = url
                doc.metadata['title'] = title
            
            # Split documents into chunks
            chunked_documents = self.text_splitter.split_documents(documents)
            
            return chunked_documents
            
        except Exception as e:
            print(f"Error processing web content {url}: {e}")
            return []
    
    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """
        Extract additional metadata from document content if possible.
        
        Args:
            document: Document object to analyze.
            
        Returns:
            Dictionary with extracted metadata.
        """
        # This is a simple implementation - you can enhance this with 
        # more sophisticated extraction techniques as needed
        metadata = {}
        
        # Try to identify publication year
        import re
        year_match = re.search(r'(19|20)\d{2}', document.page_content[:200])
        if year_match:
            metadata['year'] = year_match.group(0)
        
        # Look for keywords section
        keywords_match = re.search(r'keywords[:\s]+(.*?)(?:\.|$)', 
                                  document.page_content.lower()[:500], 
                                  re.IGNORECASE)
        if keywords_match:
            keywords = keywords_match.group(1).strip()
            metadata['keywords'] = [k.strip() for k in keywords.split(',')]
        
        return metadata
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to the directory containing PDFs.
            
        Returns:
            List of Document objects from all PDFs.
        """
        all_documents = []
        
        # Process all PDFs in the directory
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                documents = self.process_pdf(file_path)
                all_documents.extend(documents)
                
        return all_documents

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Test with a single PDF
    sample_pdf = os.path.join(config.SAMPLE_PAPERS_DIR, "sample.pdf")
    if os.path.exists(sample_pdf):
        docs = processor.process_pdf(sample_pdf)
        print(f"Processed {len(docs)} chunks from sample PDF")
    
    # Test with a web page
    sample_url = "https://en.wikipedia.org/wiki/Artificial_intelligence_in_education"
    web_docs = processor.process_web_content(sample_url)
    print(f"Processed {len(web_docs)} chunks from web page")
