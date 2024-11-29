import arxiv

def fetch_arxiv_abstracts(category, start_year, end_year, max_results):
    """
    Fetches metadata and abstracts of research papers from arXiv based on a specified category, date range, and result limit.

    Args:
        category (str): The arXiv category to search (e.g., "cs.LG" for Machine Learning).
        start_year (int): The starting year of the date range for the search (e.g., 2020).
        end_year (int): The ending year of the date range for the search (e.g., 2023).
        max_results (int): The maximum number of results to retrieve.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains metadata for a paper:
            - 'id' (str): The unique identifier for the paper on arXiv.
            - 'title' (str): The title of the paper.
            - 'published_date' (datetime.date): The publication date of the paper.
            - 'authors' (str): A comma-separated string of the authors' names.
            - 'category' (str): The category the paper belongs to.
            - 'abstract' (str): The abstract of the paper.

    """
    import arxiv

    date_query = f"submittedDate:[{start_year}0101 TO {end_year}1231]"  # Construct date query for arXiv
    search = arxiv.Search(query=f"cat:{category} AND {date_query}", max_results=max_results)  # Define search parameters

    paper_metadata = []

    try:
        results = search.results()  # Execute the search query
        if not results:
            print(f"No results for category: {category}, year: {start_year}-{end_year}")
            return paper_metadata

        # Retrieve metadata for each result
        for result in results:
            try:
                # Collect metadata including abstract
                paper_metadata.append({
                    'id': result.entry_id,
                    'title': result.title,
                    'published_date': result.published.date(),
                    'authors': ", ".join([author.name for author in result.authors]),
                    'category': category,
                    'abstract': result.summary  # Collect the abstract
                })

            except Exception as e:
                print(f"Error processing paper {result.title}: {e}")

    except Exception as e:
        print(f"Search failed: {e}")

    return paper_metadata
