import google.generativeai as genai
import creds
import re
import json 
import time
from hdbscan import HDBSCAN
from tqdm.notebook import tqdm
import plotly.graph_objects as go

GEMINI_API_KEY = creds.GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

llm_cache = {} 

def generate_content_cached(prompt, model_name="gemini-1.5-flash"):
    """
    Generates content using a generative model with caching to avoid redundant calls.

    """
    # Check if the prompt is already cached
    if prompt in llm_cache:
        # If found, retrieve the cached output to save computation time
        output = llm_cache[prompt]
    else:
        # If not cached, wait for 6 seconds (e.g., to adhere to rate limits or for model readiness)
        time.sleep(6)
        
        # Initialize the generative model using the specified model name
        gemini_model = genai.GenerativeModel(model_name)
        
        # Generate content based on the prompt and extract the text output
        output = gemini_model.generate_content(prompt).text
        
        # Store the generated output in the cache for future use
        llm_cache[prompt] = output
    
    # Return the generated output
    return output

def summarize_text(text):
    """
    Generates a title and summary for a given cluster of research abstracts.
    Cleans and parses the output to ensure a valid JSON object.
    """
    
    template = """
    You are an advanced AI system specializing in understanding and summarizing research content. Your task is to analyze the following research abstracts, which belong to a specific cluster, and provide the following:

    1. A concise and descriptive **title** that reflects the primary theme or topic shared across the abstracts.
    2. A clear and coherent **summary** that:
       - Highlights the main themes or focus areas shared across the abstracts.
       - Identifies key insights, methods, or conclusions that appear frequently.
       - Captures the overall purpose or findings of the cluster in a few sentences.

    Here are the research abstracts from the cluster:
    {text}

    Your output must strictly be a valid JSON object without any additional formatting, backticks, or newlines. Ensure it follows this exact structure:
    {{
        "title": "<Your descriptive title here>",
        "summary": "<Your detailed summary here>"
    }}
    """
    # Format the prompt with the input text
    try:
        prompt = template.format(text=text)
    except KeyError as e:
        raise ValueError(f"Error formatting template: {e}")

    # Generate the content using the cached function
    output = generate_content_cached(prompt)
    return output


def create_clusters(df_with_embeddings):
    """
    Creates clusters using the HDBSCAN algorithm based on UMAP embeddings.
    
    """
    # Initialize the HDBSCAN clustering model
    # - min_cluster_size: Minimum number of points to form a cluster.
    # - metric: Distance metric used for clustering ('euclidean' in this case).
    # - cluster_selection_method: 'eom' selects clusters using the excess of mass approach.

    if not all(col in df_with_embeddings.columns for col in ['Umap_1', 'Umap_2', 'Umap_3']):
        raise ValueError("The DataFrame must contain the columns 'Umap_1', 'Umap_2', and 'Umap_3' to perform clustering.")

    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
    
    # Fit the HDBSCAN model on the specified UMAP embedding columns and generate cluster labels
    labels = hdbscan_model.fit_predict(df_with_embeddings[['Umap_1', 'Umap_2', 'Umap_3']])

    # Return the cluster labels, where -1 indicates points classified as noise
    return labels


def clusters_with_summary(df_with_embeddings, data_type='abstract'):
    """
    Processes a DataFrame to create topic clusters for research papers, calculate centroids, and generate summaries and titles for each cluster.
    
    Args:
        df_with_embeddings (pd.DataFrame): A DataFrame containing UMAP embeddings and content ('pdf_content' or 'abstract').
        data_type (str): Specifies whether the data includes 'paper' (requires 'pdf_content') or 'abstract' (requires 'abstract').
        
    Returns:
        pd.DataFrame: A DataFrame of centroids with cluster information, including summaries and titles.
    """
    import json

    # Validate the data_type input
    if data_type not in ['paper', 'abstract']:
        raise ValueError("data_type must be either 'paper' or 'abstract'.")
    
    # Check for required columns based on the data_type
    if data_type == 'paper' and 'pdf_content' not in df_with_embeddings.columns:
        raise ValueError("The 'pdf_content' column is required in the DataFrame when data_type is 'paper'.")
    if data_type == 'abstract' and 'abstract' not in df_with_embeddings.columns:
        raise ValueError("The 'abstract' column is required in the DataFrame when data_type is 'abstract'.")

    # Step 1: Create clusters
    labels = create_clusters(df_with_embeddings)

    # Add cluster labels to the DataFrame
    df_with_embeddings['Cluster'] = labels

    # Filter out noise points (clusters labeled as -1 by HDBSCAN)
    df_with_embeddings = df_with_embeddings[df_with_embeddings['Cluster'] != -1]

    # Step 2: Calculate centroids for each cluster
    centroids = df_with_embeddings.groupby('Cluster')[['Umap_1', 'Umap_2', 'Umap_3']].mean().reset_index()

    # Calculate cluster counts (number of points in each cluster)
    cluster_counts = df_with_embeddings['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']

    # Merge centroids with cluster counts
    centroids = centroids.merge(cluster_counts, on='Cluster')

    # Step 3: Generate summaries and titles for each cluster
    summaries = []  # List to store JSON summaries

    for cluster_label in tqdm(df_with_embeddings['Cluster'].unique(), desc="Processing clusters", leave=True):
        # Filter data points for the current cluster
        cluster = df_with_embeddings[df_with_embeddings['Cluster'] == cluster_label]

        # Extract relevant text based on the data_type
        if data_type == 'paper':
            text = " ".join(cluster['pdf_content'])
        elif data_type == 'abstract':
            text = " ".join(cluster['abstract'])

        # Generate a summary and title for the current cluster
        cluster_summary_with_title = summarize_text(text)

        # Debugging: Log the raw output
        print(f"Raw Output for Cluster {cluster_label}:\n{cluster_summary_with_title}")

        # Enhanced cleaning to remove artifacts
        if isinstance(cluster_summary_with_title, str):
            cluster_summary_with_title = cluster_summary_with_title.strip()  # Remove leading/trailing whitespace
            cluster_summary_with_title = cluster_summary_with_title.lstrip('```json').rstrip('```')  # Remove backticks and json marker
            cluster_summary_with_title = cluster_summary_with_title.strip()  # Strip again after cleaning

        # Debugging: Log the cleaned output
        print(f"Cleaned Output for Cluster {cluster_label}:\n{cluster_summary_with_title}")

        # Parse the cleaned JSON string
        try:
            cluster_summary_with_title = json.loads(cluster_summary_with_title)
        except json.JSONDecodeError as e:
            print(f"Error processing cluster {cluster_label}: {e}")
            continue

        summaries.append(cluster_summary_with_title)

    # Step 4: Add summaries and titles to the centroids DataFrame
    centroids['Summary'] = [summary['summary'] for summary in summaries]
    centroids['Title'] = [summary['title'] for summary in summaries]

    # Return the centroids DataFrame with added summaries and titles
    return df_with_embeddings, centroids


def visualize(df_with_embeddings, centroids_df):

    if 'Summary' not in centroids_df.columns and 'Title' not in centroids_df.columns:
        raise ValueError("centroids_df DataFrame must contain both 'Summary' and 'Title' columns to proceed. Ensure that the clustering process includes generated summaries and titles.")
    
    if not all(col in df_with_embeddings.columns for col in ['Umap_1', 'Umap_2', 'Umap_3']):
        raise ValueError("df_with_embeddings DataFrame must contain the columns 'Umap_1', 'Umap_2', and 'Umap_3' to perform clustering.")

    centroids_df['Size'] = (centroids_df['Count'] ** 0.5) / (centroids_df['Count'].max() ** 0.5) * 100

    

