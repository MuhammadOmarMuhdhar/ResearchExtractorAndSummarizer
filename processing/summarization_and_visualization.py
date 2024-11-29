import google.generativeai as genai
from config import creds
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

    if not all(col in df_with_embeddings.columns for col in ['Umap_1', 'Umap_2', 'Umap_3']):
        raise ValueError("The DataFrame must contain the columns 'Umap_1', 'Umap_2', and 'Umap_3' to perform clustering.")

    hdbscan_model = HDBSCAN(min_cluster_size=5,max_cluster_size=50, metric='euclidean', cluster_selection_method='eom')
    
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

    for cluster_label in tqdm(centroids['Cluster'].unique(), desc="Processing clusters", leave=True):
        # Filter data points for the current cluster
        cluster = df_with_embeddings[df_with_embeddings['Cluster'] == cluster_label]

        # Extract relevant text based on the data_type
        if data_type == 'paper':
            text = " ".join(cluster['pdf_content'])
        elif data_type == 'abstract':
            text = " ".join(cluster['abstract'])

        # Generate a summary and title for the current cluster
        cluster_summary_with_title = summarize_text(text)

        # # Debugging: Log the raw output
        # print(f"Raw Output for Cluster {cluster_label}:\n{cluster_summary_with_title}")

        # Enhanced cleaning to remove artifacts
        if isinstance(cluster_summary_with_title, str):
            cluster_summary_with_title = cluster_summary_with_title.strip()  # Remove leading/trailing whitespace
            cluster_summary_with_title = cluster_summary_with_title.lstrip('```json').rstrip('```')  # Remove backticks and json marker
            cluster_summary_with_title = cluster_summary_with_title.strip()  # Strip again after cleaning

        # # Debugging: Log the cleaned output
        # print(f"Cleaned Output for Cluster {cluster_label}:\n{cluster_summary_with_title}")

        # Parse the cleaned JSON string
        try:
            cluster_summary_with_title = json.loads(cluster_summary_with_title)
        except json.JSONDecodeError as e:
            # print(f"Error processing cluster {cluster_label}: {e}")
            continue

        summaries.append(cluster_summary_with_title)

    # Step 4: Add summaries and titles to the centroids DataFrame
    centroids['Summary'] = [summary['summary'] for summary in summaries]
    centroids['Title'] = [summary['title'] for summary in summaries]

    # Return the centroids DataFrame with added summaries and titles
    return df_with_embeddings, centroids


def visualize(df_with_embeddings, centroids_df):

    """
    Generates a 3D scatter plot to visualize research documents and their topic clusters based on UMAP embeddings.
    The function validates input data, calculates cluster sizes, and creates an interactive Plotly visualization
    with options to toggle between document points, topic clusters, and combined views.

    Args:
        df_with_embeddings (pd.DataFrame): A DataFrame containing UMAP embeddings for individual research documents
                                           with columns ['Umap_1', 'Umap_2', 'Umap_3'], and cluster labels.
        centroids_df (pd.DataFrame): A DataFrame containing the cluster centroids with additional metadata, 
                                     including the required columns ['Umap_1', 'Umap_2', 'Umap_3', 'Count', 'Title', 'Summary'].

    Returns:
        None: Displays an interactive 3D scatter plot in the browser or notebook.

    Raises:
        ValueError: If `centroids_df` does not contain the required columns ['Summary', 'Title'].
        ValueError: If `df_with_embeddings` does not contain the required columns ['Umap_1', 'Umap_2', 'Umap_3'].
    """

    if 'Summary' not in centroids_df.columns and 'Title' not in centroids_df.columns:
        raise ValueError("centroids_df needs to be a DataFrame must contain both 'Summary' and 'Title' columns to proceed. Ensure that the clustering process includes generated summaries and titles.")
    
    if not all(col in df_with_embeddings.columns for col in ['Umap_1', 'Umap_2', 'Umap_3']):
        raise ValueError("df_with_embeddings needs to be a DataFrame must contain the columns 'Umap_1', 'Umap_2', and 'Umap_3' to perform clustering.")

    centroids_df['Size'] = (centroids_df['Count'] ** 0.5) / (centroids_df['Count'].max() ** 0.5) * 50

        # Create the base 3D scatter plot
    fig = go.Figure()

    # Add points from the dataset with default opacity
    fig.add_trace(go.Scatter3d(
        x=df_with_embeddings['Umap_1'],
        y=df_with_embeddings['Umap_2'],
        z=df_with_embeddings['Umap_3'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_with_embeddings['Cluster'],  # Cluster-based coloring
            colorscale='Viridis',  # Color scale for clusters
            opacity=0.8,  # Default opacity for documents

        ),
        name='Document Points)',
        visible=False,  # Default visibility
        hovertext=(
            "<b>Title:</b> " + df_with_embeddings['title'] 
            # + "<br><b>Abstract:</b> " + df_with_embeddings['abstract']
            ),
        hoverinfo="text"
    ))

    # Add cluster centroids with size scaled by density
    fig.add_trace(go.Scatter3d(
        x=centroids_df['Umap_1'],
        y=centroids_df['Umap_2'],
        z=centroids_df['Umap_3'],
        mode='markers',
        marker=dict(
            size=centroids_df['Size'],  # Use scaled size for centroids
            color=centroids_df['Count'],  # Use the number of documents as the color value
            colorscale='Picnic',  # Sequential scale for density
            symbol='circle',
            colorbar=dict(
                title="Density",
                thickness=10,
                len=0.5,
                
            ),
            opacity=.5  # Default opacity for clusters
        ),
        name='Topic Clusters (Density-Based Size and Color)',
        hovertext=(
            "<b>Topic:</b> " + centroids_df['Title'] 
            # + "<br><b>Summary:</b> " + 
            # centroids['Summary']
            ),
        hoverinfo="text", 
        visible=True  # Default visibility
    ))

    # Add cluster centroids with lower opacity for alternative views
    fig.add_trace(go.Scatter3d(
        x=centroids_df['Umap_1'],
        y=centroids_df['Umap_2'],
        z=centroids_df['Umap_3'],
        mode='markers',
        marker=dict(
            size=centroids_df['Size'],  # Use scaled size for centroids
            color=centroids_df['Count'],  # Use the number of documents as the color value
            colorscale='Picnic',  # Sequential scale for density
            symbol='circle',
            colorbar=dict(
                title="Density",
                thickness=10,
                len=0.5,
            
            ),
            opacity=0.3  # Lower opacity for this layer
        ),
        name='Cluster Centroids (Density-Based Coloring)',
        visible=False, # Default visibility
        hoverinfo="none" 
    ))

    # Add buttons to toggle visibility
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label="Topic Clusters",
                        method="update",
                        args=[
                            {"visible": [False, True, False]},  # Hide: Documents, Show: Default Centroids
                            {"title": "Scatterplot of Labeled Topic Clusters"}
                        ]
                    ),
                    dict(
                        label="Research Titles and Topic Clusters",
                        method="update",
                        args=[
                            {"visible": [True, False, True]},  # Show: Documents, Low Opacity Centroids
                            {"title": "Scatterplot of Labeled Research Titles and Topic Clusters"}
                        ]
                    ),
                    dict(
                        label="Research Titles",
                        method="update",
                        args=[
                            {"visible": [True, False, False]},  # Show: Documents, Hide: Centroids
                            {"title": "Scatterplot of Labeled Research Titles"}
                        ]
                    ),
                    
                ],
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="left",
                y=1,
                yanchor="bottom"
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='',
                showgrid=True,
                gridcolor='lightgray',
                gridwidth=2,
                showticklabels=False,
                dtick=2,
                backgroundcolor='white'
            ),
            yaxis=dict(
                title='',
                showgrid=False,
                gridcolor='lightgray',
                gridwidth=2,
                showticklabels=False,
                backgroundcolor='white'
            ),
            zaxis=dict(
                title='',
                showgrid=False,
                gridcolor='lightgray',
                gridwidth=2,
                showticklabels=False,
                backgroundcolor='white'
            ),
        ),
        paper_bgcolor='white',
        width=1150,
        height=850,
        title="Scatterplot of Labeled Topic Clusters"
    )

    # Show the plot
    return fig.show()




        

