"""
Retrieval module for the AI Education Evidence Library.
Handles the retrieval of relevant documents for a given query.
"""
from typing import List, Dict, Any, Optional
from langchain.schema.document import Document
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

from src.vector_store import VectorStore
import config

class DocumentRetriever:
    """Retrieves relevant documents for a query."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the document retriever.
        
        Args:
            vector_store: Vector store for document retrieval.
        """
        # Initialize vector store if not provided
        if vector_store is None:
            self.vector_store = VectorStore()
        else:
            self.vector_store = vector_store
            
        # Create the base retriever
        self.base_retriever = self.vector_store.vector_db.as_retriever(
            search_kwargs={"k": config.MAX_DOCUMENTS}
        )
        
        # Create the enhanced retriever with filtering
        self._setup_retriever()
    
    def _setup_retriever(self):
        """Set up the retrieval pipeline with compression."""
        # Create embeddings filter to refine results
        embeddings_filter = EmbeddingsFilter(
            embeddings=self.vector_store.embeddings, 
            similarity_threshold=0.7
        )
        
        # Create compression retriever
        self.retriever = ContextualCompressionRetriever(
            base_retriever=self.base_retriever,
            doc_compressor=embeddings_filter
        )
    
    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query text.
            k: Number of documents to retrieve (overrides config if provided).
            
        Returns:
            List of relevant Document objects.
        """
        if k is None:
            k = config.MAX_DOCUMENTS
            
        try:
            return self.retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"Error in retrieve_documents: {e}")
            # Fall back to base retriever if compression fails
            return self.base_retriever.get_relevant_documents(query)
    
    def retrieve_with_metadata_filter(self, query: str, 
                                     filter_dict: Dict[str, Any], 
                                     k: int = None) -> List[Document]:
        """
        Retrieve documents with metadata filtering.
        
        Args:
            query: The query text.
            filter_dict: Dictionary of metadata filters.
            k: Number of documents to retrieve.
            
        Returns:
            List of relevant Document objects matching filters.
        """
        if k is None:
            k = config.MAX_DOCUMENTS
            
        try:
            # Create a filtered retriever
            filtered_retriever = self.vector_store.vector_db.as_retriever(
                search_kwargs={
                    "k": k * 2,  # Retrieve more initially to allow for filtering
                    "filter": filter_dict
                }
            )
            
            # Get documents
            return filtered_retriever.get_relevant_documents(query)[:k]
        except Exception as e:
            print(f"Error in retrieve_with_metadata_filter: {e}")
            # Fall back to standard retrieval
            return self.retrieve_documents(query, k)
    
    def retrieve_by_recency(self, query: str, 
                           min_year: Optional[int] = None, 
                           max_year: Optional[int] = None,
                           k: int = None) -> List[Document]:
        """
        Retrieve documents within a year range.
        
        Args:
            query: The query text.
            min_year: Minimum publication year.
            max_year: Maximum publication year.
            k: Number of documents to retrieve.
            
        Returns:
            List of relevant Document objects within year range.
        """
        # Build filter dict
        filter_dict = {}
        if min_year is not None:
            filter_dict["year"] = {"$gte": str(min_year)}
        if max_year is not None:
            if "year" in filter_dict:
                filter_dict["year"]["$lte"] = str(max_year)
            else:
                filter_dict["year"] = {"$lte": str(max_year)}
                
        # Use metadata filter if years specified, otherwise standard retrieval
        if filter_dict:
            return self.retrieve_with_metadata_filter(query, filter_dict, k)
        else:
            return self.retrieve_documents(query, k)

# Example usage
if __name__ == "__main__":
    # Initialize the retriever
    retriever = DocumentRetriever()
    
    # Test with a query
    query = "What are the barriers to LMICs building their own AI models for education?"
    documents = retriever.retrieve_documents(query, k=3)
    
    # Print results
    print(f"Retrieved {len(documents)} documents for query: '{query}'")
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:150]}...")
