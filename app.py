"""
Streamlit application for the AI Education Evidence Library.
Allows users to search the evidence library and get answers with citations.
"""
import os
import streamlit as st
from datetime import datetime
import time

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.qa_chain import QAChain
import config

# Page configuration
st.set_page_config(
    page_title="AI Education Evidence Library",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize and cache the main components."""
    doc_processor = DocumentProcessor()
    vector_store = VectorStore()
    qa_chain = QAChain(vector_store=vector_store)
    return doc_processor, vector_store, qa_chain

doc_processor, vector_store, qa_chain = initialize_components()

# Sidebar with library stats and options
with st.sidebar:
    st.title("📚 AI Education Evidence Library")
    
    # Get and display library stats
    stats = vector_store.get_collection_stats()
    st.metric("Documents in Library", stats["count"])
    
    st.divider()
    
    # Admin section (collapsible)
    with st.expander("Admin Tools"):
        # File uploader for adding documents
        uploaded_file = st.file_uploader("Add PDF to Library", type="pdf")
        if uploaded_file:
            # Save the file temporarily
            temp_path = os.path.join(config.SAMPLE_PAPERS_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file
            with st.spinner("Processing document..."):
                documents = doc_processor.process_pdf(temp_path)
                if documents:
                    vector_store.add_documents(documents)
                    st.success(f"Added {len(documents)} chunks from {uploaded_file.name}")
                else:
                    st.error("Failed to process document")
        
        # Web content ingestion
        st.divider()
        web_url = st.text_input("Add Web Content (URL)")
        if web_url and st.button("Process URL"):
            with st.spinner("Processing web content..."):
                documents = doc_processor.process_web_content(web_url)
                if documents:
                    vector_store.add_documents(documents)
                    st.success(f"Added {len(documents)} chunks from web content")
                else:
                    st.error("Failed to process web content")
    
    # About section
    st.divider()
    st.markdown("""
    ### About
    This is a natural language evidence library focused on AI for education research,
    with special emphasis on challenges and opportunities for Low and Middle Income Countries (LMICs).
    
    Ask questions in natural language and get answers based on the research in our collection.
    """)

# Main content area
st.title("AI for Education Evidence Library")
st.write("Ask questions about AI in education research and get evidence-based answers with citations.")

# Search bar
query = st.text_input(
    "Ask a question about AI in education",
    placeholder="e.g., What are the barriers to LMICs building their own AI models?"
)

# Execute search
if query:
    # Check if the QA chain is properly initialized
    if not qa_chain.chain:
        st.error("QA system not fully initialized. Please check your API key configuration.")
    else:
        with st.spinner("Searching the evidence library..."):
            # Track time for response generation
            start_time = time.time()
            
            # Get the answer
            result = qa_chain.answer_question(query)
            
            # Calculate response time
            response_time = time.time() - start_time
            
        # Display the answer
        st.markdown("### Answer")
        st.markdown(result["answer"])
        
        # Display metadata about the response
        st.caption(f"Response generated in {response_time:.2f} seconds")
        
        # Display sources
        if result["sources"]:
            st.markdown("### Sources")
            
            for i, source in enumerate(result["sources"]):
                with st.expander(f"{i+1}. {source['title']}"):
                    st.markdown(f"**Source:** {source['source']}")
                    st.markdown("**Preview:**")
                    st.markdown(source["snippet"])
        else:
            st.info("No specific sources were cited for this answer.")

# Add example questions to help users get started
if not query:
    st.markdown("### Example Questions")
    example_questions = [
        "What are the barriers to LMICs building their own AI models for education?",
        "How can AI address educational challenges in resource-constrained environments?",
        "What are the ethical concerns of implementing AI in education in developing countries?",
        "What successful case studies exist of AI educational tools in Africa?"
    ]
    
    for question in example_questions:
        if st.button(question):
            # This will trigger the app to rerun with the selected question
            st.text_input(
                "Ask a question about AI in education",
                value=question,
                key="query_input"
            )

# Footer
st.divider()
st.caption(f"AI Education Evidence Library - Last updated: {datetime.now().strftime('%Y-%m-%d')}")
