from sentence_transformers import SentenceTransformer
import pandas as pd 
from sklearn.decomposition import PCA
from umap import UMAP


def get_embeddings(text):
    """
    Generates embeddings for a given text using the SentenceTransformer model.

    """
    # Load the pre-trained SentenceTransformer model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings for the input text
    embeddings = embedding_model.encode(text)
    
    # Convert the embeddings into a DataFrame for structured use
    embeddings_df = pd.DataFrame(embeddings)

    return embeddings_df

def optimal_pca_components(embeddings, variance_threshold=0.80):
    """
    Determines the minimum number of PCA components required to capture the specified variance.
    Fits the PCA model and transforms the data accordingly.

    """
    # Initialize PCA without specifying the number of components
    pca = PCA()
    pca.fit(embeddings)
    
    # Compute cumulative explained variance ratio
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    
    # Find the number of components that meet or exceed the threshold
    n_components = (cumulative_variance >= variance_threshold).argmax() + 1
    
    # Fit a new PCA model with the optimal number of components
    pca_model = PCA(n_components=n_components)
    pca_transformed = pca_model.fit_transform(embeddings)

    pca_df = pd.DataFrame(pca_transformed, columns=[f"PC{i+1}" for i in range(pca_model.n_components_)])
    
    return pca_df

def umap_transformation(pca_embeddings):
    """
    Applies UMAP dimensionality reduction to transform PCA embeddings into 3D space.
    
    """
    # Perform UMAP transformation
    embedding_2d = UMAP(random_state=211).fit_transform(pca_embeddings)
    embedding_df_2d = pd.DataFrame(embedding_2d, columns=['Umap_1', 'Umap_2'])
    return embedding_df_2d



def reduced_embeddings(data_frame, data_type):
    """
    Generates reduced embeddings from a DataFrame based on the specified data type (either 'paper' or 'abstract').

    Parameters:
        data_frame (DataFrame): Input DataFrame containing text data.
        data_type (str): Specifies the type of data ('paper' or 'abstract').

    Returns:
        DataFrame: A DataFrame with original data and reduced embeddings.
    """
    # Validate that data_type is either 'paper' or 'abstract'
    if data_type not in ['paper', 'abstract']:
        raise ValueError("data_type must be either 'paper' or 'abstract'.")

    # Validate that the necessary column exists for the selected data_type
    if data_type == 'paper' and 'pdf_content' not in data_frame.columns:
        raise ValueError("The 'pdf_content' column is required in the DataFrame when data_type is 'paper'.")
    if data_type == 'abstract' and 'abstract' not in data_frame.columns:
        raise ValueError("The 'abstract' column is required in the DataFrame when data_type is 'abstract'.")

    # Select the appropriate column based on data_type
    if data_type == 'paper':
        text = data_frame['pdf_content']
    else:
        text = data_frame['abstract']

    # Generate embeddings and reduce dimensions
    embeddings = get_embeddings(text)
    pca_reduced_embeddings = optimal_pca_components(embeddings)
    embedding_df_2d = umap_transformation(pca_reduced_embeddings)

    # Concatenate the original data with reduced embeddings
    df = pd.concat([data_frame.reset_index(drop=True), embedding_df_2d.reset_index(drop=True)], axis=1)

    return df





