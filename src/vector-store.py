"""
Vector store module for the AI Education Evidence Library.
Handles creating and querying the document vector database.
"""
from typing import List, Dict, Any, Optional
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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
            persist_directory = config.VECTOR_STORE_PATH
            
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        
        # Initialize or load the vector store
        if os.path.exists(persist_directory):
            self.vector_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Loaded existing vector store from {persist_directory}")
        else:
            self.vector_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            print(f"Created new vector store at {persist_directory}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add.
        """
        try:
            self.vector_db.add_documents(documents)
            self.vector_db.persist()
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
            return {
                "count": self.vector_db._collection.count(),
                "metadata_keys": list(self.vector_db._collection.get()["metadatas"][0].keys())
                if self.vector_db._collection.count() > 0 else []
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {"count": 0, "metadata_keys": []}

# Example usage
if __name__ == "__main__":
    # Initialize vector store
    vector_store = VectorStore()
    
    # Print stats
    stats = vector_store.get_collection_stats()
    print(f"Vector store contains {stats['count']} documents")
    print(f"Available metadata fields: {stats['metadata_keys']}")
    
    # Test search if documents exist
    if stats['count'] > 0:
        results = vector_store.similarity_search(
            "What are the challenges of AI in education?", 
            k=2
        )
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Content: {doc.page_content[:200]}...")
