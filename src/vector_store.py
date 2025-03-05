"""
Vector store module for the AI Education Evidence Library.
Handles creating and querying the document vector database.
"""
from typing import List, Dict, Any, Optional
import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings  # Use OpenAI embeddings instead
from langchain.schema.document import Document

import config

class VectorStore:
    """Vector store for document embeddings and retrieval."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector store.
                             If None, uses the default from config.
        """
        if persist_directory is None:
            self.persist_directory = config.VECTOR_STORE_PATH
        else:
            self.persist_directory = persist_directory
            
        # Initialize OpenAI embeddings instead of HuggingFace
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Most cost-effective OpenAI embedding model
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Rest of the code remains the same
        self.index_file = os.path.join(self.persist_directory, "faiss_index")
        self.docs_file = os.path.join(self.persist_directory, "faiss_docs")
        
        if os.path.exists(self.index_file) and os.path.exists(self.docs_file):
            try:
                self.vector_db = FAISS.load_local(
                    self.persist_directory,
                    self.embeddings,
                    "faiss_index"
                )
                print(f"Loaded existing vector store from {self.persist_directory}")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vector_db = FAISS.from_documents(
                    [Document(page_content="Placeholder", metadata={})],
                    self.embeddings
                )
        else:
            # Create directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize with a placeholder document to create the structure
            self.vector_db = FAISS.from_documents(
                [Document(page_content="Placeholder", metadata={})],
                self.embeddings
            )
            self.vector_db.save_local(self.persist_directory, "faiss_index")
            print(f"Created new vector store at {self.persist_directory}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add.
        """
        try:
            if len(documents) > 0:
                # Add documents to the vector store
                self.vector_db.add_documents(documents)
                
                # Save the updated vector store
                self.vector_db.save_local(self.persist_directory, "faiss_index")
                print(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search for a query.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            
        Returns:
            List of relevant Document objects.
        """
        try:
            return self.vector_db.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: The search query.
            k: Number of documents to retrieve.
            
        Returns:
            List of (Document, score) tuples.
        """
        try:
            return self.vector_db.similarity_search_with_score(query, k=k)
        except Exception as e:
            print(f"Error in similarity search with score: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with vector store statistics.
        """
        try:
            # For FAISS, we don't have direct collection stats like ChromaDB
            # So we'll estimate based on the index
            doc_count = len(self.vector_db.docstore._dict)
            
            # Get a sample document to extract metadata keys
            metadata_keys = []
            if doc_count > 0:
                # Get first document's metadata keys
                sample_doc_id = next(iter(self.vector_db.docstore._dict))
                sample_doc = self.vector_db.docstore._dict[sample_doc_id]
                metadata_keys = list(sample_doc.metadata.keys())
            
            return {
                "count": doc_count,
                "metadata_keys": metadata_keys
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"count": 0, "metadata_keys": []}
