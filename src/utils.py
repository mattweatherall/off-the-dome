"""
Utility functions for the AI Education Evidence Library.
"""
import os
import re
from typing import Dict, List, Any, Optional
import requests
from bs4 import BeautifulSoup

def extract_citation_from_text(text: str) -> Dict[str, List[str]]:
    """
    Extract citation references from generated text.
    
    Args:
        text: The text to extract citations from.
        
    Returns:
        Dictionary mapping source IDs to citation contexts.
    """
    # Look for citations in format [Source: Title]
    citation_pattern = r'\[Source:\s+([^\]]+)\]'
    citations = {}
    
    for match in re.finditer(citation_pattern, text):
        source = match.group(1).strip()
        
        # Get some context (50 chars before and after)
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end]
        
        if source not in citations:
            citations[source] = []
        
        citations[source].append(context)
    
    return citations

def download_sample_papers() -> List[str]:
    """
    Download some sample papers to get started.
    
    Returns:
        List of file paths to downloaded papers.
    """
    # List of sample papers (URLs to PDFs)
    sample_papers = [
        "https://unesdoc.unesco.org/ark:/48223/pf0000376709/PDF/376709eng.pdf.multi",  # UNESCO AI in education
        "https://www.ece.gov.nt.ca/sites/ece/files/resources/unesco_ai_in_education.pdf"  # Another UNESCO report
    ]
    
    downloaded_files = []
    
    for url in sample_papers:
        try:
            # Extract filename from URL
            filename = os.path.basename(url.split('?')[0])
            if not filename.endswith('.pdf'):
                filename = f"sample_{len(downloaded_files)}.pdf"
                
            filepath = os.path.join(os.environ.get('SAMPLE_PAPERS_DIR', './data/sample_papers'), filename)
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            downloaded_files.append(filepath)
            print(f"Downloaded {filename}")
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    
    return downloaded_files

def fetch_papers_from_google_scholar(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch paper information from Google Scholar.
    Note: This is a very simple implementation and may not work reliably
    due to Google Scholar's anti-scraping measures.
    
    Args:
        query: Search query.
        num_results: Number of results to fetch.
        
    Returns:
        List of paper information dictionaries.
    """
    # This is a placeholder implementation
    # For a production system, consider using a proper API or service
    
    # Build the search URL
    base_url = "https://scholar.google.com/scholar"
    params = {
        "q": query,
        "hl": "en",
        "num": num_results
    }
    
    papers = []
    
    try:
        # Set a user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Make the request
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract paper information
        for result in soup.select('.gs_ri'):
            try:
                title_elem = result.select_one('.gs_rt')
                title = title_elem.text if title_elem else "Unknown Title"
                
                link = None
                if title_elem and title_elem.select_one('a'):
                    link = title_elem.select_one('a')['href']
                
                authors_year_elem = result.select_one('.gs_a')
                authors_year = authors_year_elem.text if authors_year_elem else ""
                
                # Try to extract year
                year_match = re.search(r'(\d{4})', authors_year)
                year = year_match.group(1) if year_match else None
                
                # Try to extract authors
                authors = authors_year.split('-')[0].strip() if '-' in authors_year else authors_year
                
                # Extract snippet
                snippet_elem = result.select_one('.gs_rs')
                snippet = snippet_elem.text if snippet_elem else ""
                
                papers.append({
                    "title": title,
                    "link": link,
                    "authors": authors,
                    "year": year,
                    "snippet": snippet
                })
            except Exception as e:
                print(f"Error parsing a result: {e}")
        
        return papers
        
    except Exception as e:
        print(f"Error fetching from Google Scholar: {e}")
        return []

def format_citation(metadata: Dict[str, Any]) -> str:
    """
    Format metadata into a citation string.
    
    Args:
        metadata: Document metadata.
        
    Returns:
        Formatted citation string.
    """
    title = metadata.get('title', 'Unknown Title')
    authors = metadata.get('authors', metadata.get('author', 'Unknown Author'))
    year = metadata.get('year', 'n.d.')
    source = metadata.get('source', 'Unknown Source')
    
    if isinstance(authors, list):
        authors = ', '.join(authors)
    
    return f"{authors} ({year}). {title}. {source}."
