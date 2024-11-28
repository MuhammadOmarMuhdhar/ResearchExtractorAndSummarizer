import arxiv

def collect_paper_abstracts(category, start_year, end_year, max_results):
    """
    Searches for papers on arXiv within a specified category and date range.
    Collects metadata, including abstracts, without introducing delays.
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
