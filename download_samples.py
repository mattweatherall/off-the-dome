"""
Utility script to download sample papers for the AI Education Evidence Library.
"""
import os
import sys
import requests
from pathlib import Path

# Add the project root to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore

def download_file(url, destination):
    """Download a file from URL to destination."""
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    """Main function to download and process sample papers."""
    # Ensure sample papers directory exists
    os.makedirs(config.SAMPLE_PAPERS_DIR, exist_ok=True)
    
    # List of sample papers about AI in education
    sample_papers = [
        {
            "url": "https://unesdoc.unesco.org/ark:/48223/pf0000376709/PDF/376709eng.pdf.multi",
            "filename": "unesco_ai_in_education.pdf"
        },
        {
            "url": "https://www.oecd.org/education/trustworthy-artificial-intelligence-in-education-c65a482d-en.htm",
            "filename": "oecd_trustworthy_ai_in_education.pdf"
        },
        {
            "url": "https://www.researchgate.net/profile/Santosh-Mishra-25/publication/348586471_AI_in_Education_An_Enabler_of_Inclusion_or_a_Cause_of_Exclusion_for_Students_with_Disabilities_in_Developing_Countries/links/6006c7f745851553a060f93d/AI-in-Education-An-Enabler-of-Inclusion-or-a-Cause-of-Exclusion-for-Students-with-Disabilities-in-Developing-Countries.pdf",
            "filename": "ai_education_lmics_disabilities.pdf"
        }
    ]
    
    # Download papers
    downloaded_files = []
    for paper in sample_papers:
        destination = os.path.join(config.SAMPLE_PAPERS_DIR, paper["filename"])
        if download_file(paper["url"], destination):
            downloaded_files.append(destination)
    
    # Process downloaded papers
    if downloaded_files:
        print("\nProcessing downloaded papers...")
        doc_processor = DocumentProcessor()
        vector_store = VectorStore()
        
        for file_path in downloaded_files:
            print(f"Processing {os.path.basename(file_path)}...")
            documents = doc_processor.process_pdf(file_path)
            if documents:
                vector_store.add_documents(documents)
                print(f"Added {len(documents)} chunks to vector store")
            else:
                print(f"Failed to process {file_path}")
        
        print("\nSample papers processed and added to vector store!")
        print("You can now run the application with: streamlit run app.py")
    else:
        print("\nNo papers were downloaded successfully. Please check the URLs or your internet connection.")

if __name__ == "__main__":
    main()
