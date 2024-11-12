import re 
import requests  
from io import BytesIO  
import arxiv  
import spacy  
from PyPDF2 import PdfReader  
from tqdm import tqdm
import os
import time
import random

nlp = spacy.load("en_core_web_sm")




def collect_paper_metadata(category, start_year, end_year, max_results):
    """
    Searches for papers on arXiv within a specified category and date range.
    Collects metadata and downloads PDF content as BytesIO objects using requests.
    Implements adaptive delay to handle rate limiting specifically.
    """
    date_query = f"submittedDate:[{start_year}0101 TO {end_year}1231]"  # Construct date query for arXiv
    search = arxiv.Search(query=f"cat:{category} AND {date_query}", max_results=max_results)  # Define search parameters

    paper_metadata = []
    base_delay = 3  # Start with 3 second delay
    max_delay = 15  # Maximum delay cap for rate limits
    adaptive_delay = base_delay  # Initialize adaptive delay

    try:
        results = search.results()  # Execute the search query
        if not results:
            print(f"No results for category: {category}, year: {start_year}-{end_year}")

        # Retrieve metadata and download PDF content for each result
        for result in results:
            try:
                # Attempt to fetch the PDF
                response = requests.get(result.pdf_url, stream=True)
                
                # Explicit check for rate limiting
                if response.status_code == 429:  # Rate limit status code
                    print("Rate limit reached. Increasing delay.")
                    adaptive_delay = min(adaptive_delay * 2 + 0.5, max_delay)  # Increase delay with a cap
                    time.sleep(adaptive_delay)
                    continue  # Skip the current iteration and retry later

                # If the request is successful
                response.raise_for_status()  # Raise an error for other unsuccessful status codes
                pdf_content = BytesIO(response.content)  # Store PDF as BytesIO object

                # Append metadata and PDF content to the list
                paper_metadata.append({
                    'id': result.entry_id,
                    'title': result.title,
                    'published_date': result.published.date(),
                    'authors': ", ".join([author.name for author in result.authors]),
                    'category': category,
                    'pdf_content': pdf_content  # Store BytesIO object of the PDF content
                })

                # Reset adaptive delay to 0 on success
                adaptive_delay = base_delay
                time.sleep(random.uniform(adaptive_delay, adaptive_delay + 0.5))  # Slight randomization to avoid exact intervals

            except requests.exceptions.RequestException as e:
                # Handle other types of exceptions differently if desired
                print(f"Failed to download PDF for {result.title}: {e}")
                adaptive_delay = min(adaptive_delay * 2 + 0.5, max_delay)  # Incrementally increase delay
                time.sleep(adaptive_delay)

    except Exception as e:
        print(f"Search failed: {e}")

    return paper_metadata


def clean_pdf_text(text):
    """
    Cleans extracted PDF text while preserving structure and avoiding excessive cleaning.
    """
    # Replace non-breaking spaces and merge broken word patterns
    text = text.replace('\xa0', ' ')
    text = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2', text) 
    text = re.sub(r'\b(\w)\s+(\w+)\b', r'\1\2', text)  

    # Remove hyphenated line breaks to ensure words are intact
    text = re.sub(r'(\w)-\s', r'\1', text)
    
    # Standardize spaces, especially within references
    text = re.sub(r'\[\s+(\d+)(,\s*\d+)*\s*\]', lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove unwanted characters while keeping essential punctuation
    text = re.sub(r'[^\w\s.,:;\'\"()\[\]]+', ' ', text)
    
    # Standardize quotes and dashes
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('–', '-').replace('—', '-')
    
    return text


def extract_relevant_sections(text):
    """
    Extracts sections from the text that are likely to be relevant to a high-level summary.
    Looks for headings like Introduction, Core Concept, Experimental Results, and Conclusion.
    """
    # Define patterns for identifying section headings of interest
    patterns = [
        r'(Introduction|Background)',  # Identify Introduction or background section
        r'(Core Concept|Methodology|Approach)',  # Core concept or approach section
        r'(Experimental Results|Findings|Evaluation)',  # Experimental results section
        r'(Conclusion|Future Work|Discussion)',  # Conclusion or future directions section
    ]
    
    relevant_sections = []  # Initialize list for storing relevant sections
    
    # Search for each pattern and extract up to 2000 characters after the heading
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.start()
            end = text.find("\n", start + 2000) if text.find("\n", start + 2000) != -1 else len(text)
            section_text = text[start:end]
            relevant_sections.append(section_text)
    
    return ' '.join(relevant_sections)  # Return joined relevant sections as a single text


def get_pdf_text(pdf_content):
    """
    Extracts text from all pages of a PDF file and preprocesses it to retain only relevant sections.
    Returns a dictionary with text from the first page and the full document.
    """
    reader = PdfReader(pdf_content)  # Load PDF file
    
    # Extract and clean text from the first page
    first_page_text = reader.pages[0].extract_text() or ""
    cleaned_first_page = clean_pdf_text(first_page_text)
    
    # Collect and clean text from all pages
    entire_text = ''.join(page.extract_text() or "" for page in reader.pages)
    cleaned_text = clean_pdf_text(entire_text)
    
    # Extract relevant sections from the cleaned text
    relevant_text = extract_relevant_sections(cleaned_text)
    
    return {
        'first_page': cleaned_first_page,
        'entire_text': relevant_text
    }


def extract_affiliations_dependency_parsing(first_page):
    """
    Extracts affiliations using NLP dependency parsing, focusing on organization-related keywords.
    Cleans and verifies extracted affiliations.
    """
    affiliation_keywords = ["university", "institute", "college", "department", "school", "laboratory", "centre", "center", "hospital"]
    potential_affiliations = []

    for line in first_page.splitlines():
        doc = nlp(line)
        for chunk in doc.noun_chunks:
            # Identify noun chunks that match affiliation keywords
            if any(word.lower_ in affiliation_keywords for word in chunk):
                affiliation = chunk.text.strip()
                affiliation = re.sub(r"\d+", "", affiliation)  # Remove numbers
                affiliation = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", affiliation)  # Remove emails
                affiliation = re.sub(r"\s+", " ", affiliation).strip()  # Clean extra whitespace

                if any(keyword in affiliation.lower() for keyword in affiliation_keywords):
                    potential_affiliations.append(affiliation)
    
    refined_affiliations = clean_affiliations(potential_affiliations)  # Refine extracted affiliations
    
    # Final verification for organizational entities using NLP
    verified_affiliations = []
    for affiliation in refined_affiliations:
        doc = nlp(affiliation)
        if any(ent.label_ in {"ORG", "GPE"} for ent in doc.ents):
            verified_affiliations.append(affiliation)
    
    return verified_affiliations


def clean_affiliations(affiliations):
    """
    Refines extracted affiliations by removing generic terms, unwanted patterns, and duplicates.
    """
    refined_affiliations = []
    for affiliation in affiliations:
        if len(affiliation.split()) < 2 or affiliation.lower() in ["department", "center", "school", "the university"]:
            continue  # Skip generic or single-word affiliations
        
        # Remove common unwanted patterns and clean whitespace
        affiliation = re.sub(r"\b(?:department|center|school|college|institute)\b", "", affiliation, flags=re.IGNORECASE)
        affiliation = re.sub(r"\s+", " ", affiliation).strip()  # Clean extra whitespace
        
        if len(affiliation.split()) >= 2:  # Ensure multi-word affiliation
            refined_affiliations.append(affiliation)
    
    return sorted(set(refined_affiliations))  # Return unique, sorted affiliations


def process_pdf(pdf_data):
    """
    Processes a single PDF, extracting text, cleaning content, and identifying affiliations.
    Returns a dictionary containing metadata, affiliations, and cleaned content.
    """
    pdf_text = get_pdf_text(pdf_data['pdf_content'])  # Extract text from PDF
    
    # Clean the first page and entire text
    first_page_cleaned = clean_pdf_text(pdf_text['first_page'])
    entire_text_cleaned = clean_pdf_text(pdf_text['entire_text'])

    # Extract and clean affiliations
    affiliations = extract_affiliations_dependency_parsing(first_page_cleaned)
    cleaned_affiliations = clean_affiliations(affiliations)

    # Return processed data
    result = {
        'id': pdf_data['id'],
        'title': pdf_data['title'],
        'published_date': pdf_data['published_date'],
        'authors': pdf_data['authors'],
        'category': pdf_data['category'],
        'affiliations': cleaned_affiliations,
        'content': entire_text_cleaned
    }

    return result

