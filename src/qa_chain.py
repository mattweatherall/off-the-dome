"""
Question answering chain for the AI Education Evidence Library.
Uses retrieval and LLM to answer questions based on the document collection.
"""
from typing import List, Dict, Any, Optional
import os

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from src.vector_store import VectorStore
import config

# Define a custom prompt template for education AI questions
PROMPT_TEMPLATE = """
You are an expert research assistant specializing in AI for education, particularly focusing on issues relevant to Low and Middle Income Countries (LMICs).
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite your sources using [Source: Title] format after each key point.

Context:
{context}

Question: {question}

Answer (be comprehensive and include all relevant information from the sources, with proper citations):
"""

class QAChain:
    """Question answering chain using RAG architecture."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the QA chain.
        
        Args:
            vector_store: Vector store for document retrieval.
        """
        # Initialize vector store if not provided
        if vector_store is None:
            self.vector_store = VectorStore()
        else:
            self.vector_store = vector_store
            
        # Initialize LLM
        # For V1, we're using a simple OpenAI setup, but this can be replaced
        # with other models like Ollama, Hugging Face models, etc.
        # The API key should be set in your environment or .env file
        if config.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                temperature=0,
                model_name=config.LLM_MODEL,
                streaming=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
            
            # Create the retrieval chain
            self._setup_chain()
        else:
            self.llm = None
            self.chain = None
            print("Warning: OPENAI_API_KEY not set. QA chain not fully initialized.")
    
    def _setup_chain(self):
        """Set up the retrieval and QA chain."""
        # Create the retrieval component with improved relevance filtering
        retriever = self.vector_store.vector_db.as_retriever(
            search_kwargs={"k": config.MAX_DOCUMENTS}
        )
        
        # Optional: Add contextual compression for better retrieval
        # This filters out less relevant passages after initial retrieval
        embeddings_filter = EmbeddingsFilter(embeddings=self.vector_store.embeddings, 
                                            similarity_threshold=0.7)
        compression_retriever = ContextualCompressionRetriever(
            retriever=retriever,
            base_compressor=embeddings_filter
        )
        
        # Create the RAG prompt
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Set up the RAG chain
        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the document collection.
        
        Args:
            question: The question to answer.
            
        Returns:
            Dictionary with answer and source information.
        """
        if not self.chain:
            return {
                "answer": "QA system not fully initialized. Please check your API key configuration.",
                "sources": []
            }
        
        try:
            # Get the raw answer
            answer = self.chain.invoke(question)
            
            # Get the source documents for citation
            source_docs = self.vector_store.similarity_search(question, k=config.MAX_DOCUMENTS)
            
            # Extract unique sources
            sources = []
            seen_sources = set()
            for doc in source_docs:
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", os.path.basename(source))
                
                # Create a unique identifier for this source
                source_id = f"{source}_{title}"
                
                if source_id not in seen_sources:
                    sources.append({
                        "title": title,
                        "source": source,
                        "snippet": doc.page_content[:200] + "..."
                    })
                    seen_sources.add(source_id)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Error in answer_question: {e}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": []
            }

# Example usage
if __name__ == "__main__":
    # Initialize the QA chain
    qa_chain = QAChain()
    
    # Test with a sample question
    if qa_chain.chain:
        result = qa_chain.answer_question(
            "What are the barriers to LMICs building their own AI models for education?"
        )
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source['title']} ({source['source']})")
