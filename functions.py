#%% import packages
import openai
import json
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

# Initialize the OpenAI API with your API key
openai.api_key = "key"

def create_embeddings(texts, engine="text-embedding-ada-002", output_file="embeddings.csv"):
    """
    Create text embeddings for a list of input texts using the specified OpenAI engine and save them to a CSV file.

    Args:
        texts (list): A list of input texts to be embedded.
        engine (str): The OpenAI engine for text embedding.
        output_file (str): The name of the CSV file to save the embeddings.

    Returns:
        list: A list of text embeddings.
    """
    resp = openai.Embedding.create(input=texts, engine=engine)
    embeddings = [entry["embedding"] for entry in resp["data"]]

    # Save the embeddings to a CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(embeddings)

    return embeddings

def calculate_similarity_matrix(embeddings, similarity_type='dot'):
    """
    Calculate the pairwise similarity matrix based on text embeddings.

    Args:
        embeddings (list): A list of text embeddings.
        similarity_type (str): 'cosine' or 'dot', specifies the type of similarity to calculate.

    Returns:
        numpy.ndarray: A 2D numpy array representing the normalized similarity matrix scaled to 100.
    """
    embeddings = np.array(embeddings)
    if similarity_type == 'cosine':
        # Calculate cosine similarity
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(embeddings, embeddings.T) / (norm * norm.T)
   
    elif similarity_type == 'dot':
        # Calculate dot product similarity
        similarity_matrix = np.dot(embeddings, embeddings.T)
    else:
        raise ValueError("Invalid similarity_type. Use 'cosine' or 'dot'.")

    # Scale the entire similarity matrix to 100
    similarity_matrix = 100 * (similarity_matrix / np.max(similarity_matrix))

    return similarity_matrix

def hierarchical_clustering(similarity_matrix, linkage_method='ward', distance_threshold=0.2):
    """
    Perform hierarchical clustering on a similarity matrix.

    Args:
        similarity_matrix (numpy.ndarray): A 2D numpy array representing the similarity matrix.
        linkage_method (str): The linkage method for hierarchical clustering.
        distance_threshold (float): The distance threshold for forming clusters.

    Returns:
        numpy.ndarray: An array of cluster assignments.
    """

    linkage_matrix = linkage(1 - similarity_matrix / 100, method=linkage_method)
    clusters = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
    return clusters


def calculate_silhouette_scores(embeddings):
    """
    Calculate silhouette scores for different numbers of clusters (k) using K-means clustering.

    Args:
        embeddings (list): A list of text embeddings for clustering.

    Returns:
        pd.DataFrame: A DataFrame with columns 'k' and 'score', containing silhouette scores for each value of k.
    """
    cluster_results_km = pd.DataFrame(columns=['k', 'score'])
    min_clusters=2
    max_clusters=len(embeddings)-1
    data_matrix=np.vstack(embeddings)
    for k in range(min_clusters, max_clusters + 1):
        km_model = KMeans(n_clusters=k, init='k-means++', random_state=42)
        y = km_model.fit_predict(data_matrix)
        silhouette = silhouette_score(data_matrix, y)
        dic={'k': [k], 'score': [silhouette]}
        cluster_results_km=pd.concat([cluster_results_km, pd.DataFrame(dic)])    
    return cluster_results_km

def find_optimal_cluster(cluster_results):
    """
    Find the optimal number of clusters based on silhouette scores.

    Args:
        cluster_results (pd.DataFrame): A DataFrame with columns 'k' and 'score' containing silhouette scores for different k values.

    Returns:
        int: The optimal number of clusters.
    """
    cluster_results = cluster_results.reset_index(drop=True)
    optimal_cluster = cluster_results['score'].idxmax()
    optimal_cluster = cluster_results['k'].iloc[optimal_cluster]
    return optimal_cluster
     

def group_cluster(inputdata, clusters, output_filename="output.json"):
    """
    Group cluster assignments and update input data with cluster information.

    Args:
        inputdata (dict): A dictionary containing "titles" and "theme_attributes".
        clusters (list): A list of cluster assignments for each title.
        output_filename (str): The name of the output JSON file. Default is "output.json".

    Returns:
        dict: An updated input data dictionary with cluster information.
    """
    themes = list(inputdata["titles"].keys())
    ## We use this variable only for the first for loop, 
    ## and its purpose is to make the primary_theme & cluster matching.
    ## As a result of this matching (for example, cluster_theme_mappings = {2: 'theme1', 1: 'theme3'}),
    ## we create themes_as_primary_themes: (example; themes_as_primary_themes = ['theme1', 'theme1', 'theme3', 'theme1', 'theme1', 'theme3', 'theme1'] for cluster [2,2,1,2,2,1,2]
    cluster_theme_mappings = {}
    themes_as_primary_themes = []

    ## The purpose of the first for loop is to create a theme list of the same length, 
    ## but in terms of primary themes (what is a primary theme?: the first theme that matches the new number in the cluster list)
    ## with the cluster being [2, 2, 1, 2, 2, 1, 2]
    ## our new theme list will be: ['theme1', 'theme1', 'theme3', 'theme1', 'theme1', 'theme3', 'theme1']
    for theme, cluster in zip(themes, clusters):
        if cluster not in cluster_theme_mappings:
            cluster_theme_mappings[cluster] = theme
        themes_as_primary_themes.append(cluster_theme_mappings[cluster])

    original_themes = list(inputdata["theme_attributes"].keys())
    updated_theme_attributes = {}

    ## The purpose of this for loop is to compare two theme arrays and create a new dictionary (updated_theme_attributes).
    for primary_theme, original_theme in zip(themes_as_primary_themes, original_themes):
        if original_theme == primary_theme:
            updated_theme_attributes[original_theme] = {}
        else:
            updated_theme_attributes[original_theme] = {"merged": primary_theme}
    
    ## We update the 'theme_attributes' of the input data with the new (merged) dictionary (updated_theme_attributes) and write it to a JSON file.
    inputdata["theme_attributes"] = updated_theme_attributes

    with open(output_filename, 'w') as json_file:
        json.dump(inputdata, json_file, indent=4)

    return inputdata

def cluster_texts(texts, clusters):
    """
    Group texts into clusters based on cluster assignments.

    Args:
        texts (list): A list of texts.
        clusters (list): A list of cluster assignments for each text.

    Returns:
        dict: A dictionary where keys are cluster IDs, and values are lists of texts in each cluster.
    """
    # Create a dictionary to store results
    cluster_results = {}
    # Group texts into clusters
    for i, text in enumerate(texts):
        cluster_id = clusters[i]
        cluster_results.setdefault(cluster_id, []).append(text)

    return cluster_results

def cluster_similar_elements( inputdata, method='hierarchical',linkage_method='ward', distance_threshold=0.25, engine="text-embedding-ada-002"):
    """
    Cluster similar elements in a list of texts and update the input data with cluster information.

    Args:
        input_data (dict): A dictionary containing "theme_attributes" and "titles".
        linkage_method (str): The linkage method for hierarchical clustering.
        distance_threshold (float): The distance threshold for forming clusters.
        engine (str): The OpenAI engine for text embedding.

    Returns:
        dict: An updated input_data dictionary with cluster information.
    """
    # Extract the list of titles from input_data
    # Load the input data from a JSON file
    input_data=inputdata

# Extract the list of titles from the "titles" section of the input data
    theme_titles = list(input_data["titles"].values())
    
    # Create embeddings
    embeddings = create_embeddings(theme_titles, engine)
    if method == 'hierarchical':
    # Calculate similarity matrix
        similarity_matrix = calculate_similarity_matrix(embeddings)

        # Perform hierarchical clustering
        h_clusters = hierarchical_clustering(similarity_matrix, linkage_method, distance_threshold)

        return group_cluster(input_data, h_clusters,output_filename="h_output.json")
    if method == 'kmeans':
        # Calculate Silhouette scores
        cluster_results_km = calculate_silhouette_scores(embeddings)
        num_cluster = find_optimal_cluster(cluster_results_km)

        # Apply K-Means clustering
        km_model = KMeans(n_clusters=num_cluster, init='k-means++', random_state=42)
        k_clusters = km_model.fit_predict(embeddings)
        return group_cluster(input_data, k_clusters,output_filename="k_output.json")
    else:
        raise ValueError("Method should be 'hierarchical' or 'kmeans'")




