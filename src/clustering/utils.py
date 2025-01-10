import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA


def find_centroids(df, target_column):
    """Find the centroid of the embeddings for each class in the target_column"""
    return df.groupby(target_column)["embeddings"].apply(
        lambda group: np.mean(np.vstack(group), axis=0)
    )


def find_distance_to_centroids(
    df, centroids, target_column="label", embedding_column="embeddings"
):
    """
    Find the cosine distance of each row's embedding to the centroid of its class.

    Parameters:
        df (pd.DataFrame): The DataFrame containing embeddings and target classes.
        centroids (pd.Series): A series where the index is the class and the value is the centroid embedding.
        target_column (str): The column name in df representing the target classes.
        embedding_column (str): The column name in df representing the embeddings.

    Returns:
        pd.Series: A series containing the cosine distance for each row in the DataFrame.
    """
    # Convert centroids to a dictionary for efficient lookup
    centroids_dict = centroids.to_dict()

    def compute_distance(row):
        # Retrieve the centroid for the row's class
        centroid = centroids_dict[row[target_column]]
        # Compute cosine similarity
        distance = cosine_distances([row[embedding_column]], [centroid])[0][0]
        return distance

    # Apply the distance computation for each row
    return df.apply(compute_distance, axis=1)


def perform_dbscan_grid_search(embeddings, param_grid):
    """
    Perform grid search to find the best DBSCAN parameters using silhouette score.

    Parameters:
        embeddings (np.ndarray): Array of text embeddings to cluster.
        param_grid (dict): Dictionary of parameters for DBSCAN.

    Returns:
        dict: Best parameters and the corresponding silhouette score.
        DBSCAN: Fitted DBSCAN model with best parameters.
    """
    best_score = -1  # Silhouette score ranges from -1 to 1
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        # Train DBSCAN model with the current set of parameters
        model = DBSCAN(**params)
        labels = model.fit_predict(embeddings)

        # Log intermediate results
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Only calculate silhouette score if there are at least 2 clusters
        if num_clusters > 1:
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model

    if best_model is None:
        print("No valid clustering found. Adjust parameters or check the data.")
    return {"best_params": best_params, "best_score": best_score}, best_model


def cluster_and_find_new_class(
    df_outliers: pd.DataFrame, known_class_info: pd.DataFrame, logger
):
    """
    Cluster the outliers and find new classes based on the distance to known class centroids.

    :param df_outliers: DataFrame containing outlier embeddings and their cluster labels.
                        Must contain a column 'embeddings' with embeddings as numpy arrays.
    :param known_class_info: DataFrame containing known class centroids.
                             Must contain columns 'centroid' with centroids as numpy arrays.
    :return: Updated DataFrame with new columns 'new_class' and 'special_case'.
    """

    # Extract embeddings from the outlier DataFrame
    emb_outliers = np.stack(df_outliers["embeddings"].values, axis=0)

    # Initialize new columns
    df_outliers["new_class"] = False  # default
    df_outliers["special_case"] = False  # default

    # Known class centroids as a numpy array
    known_centroids = np.stack(
        known_class_info["centroid"].values, axis=0
    )  # shape (num_classes, embedding_dim)

    # Iterate over unique clusters in the outliers DataFrame
    unique_clusters = df_outliers["cluster"].unique()
    for cl_id in unique_clusters:
        if cl_id == -1:
            # DBSCAN assigns -1 to noise
            df_outliers.loc[df_outliers["cluster"] == -1, "special_case"] = True
            continue

        # Extract embeddings for the current cluster
        mask = df_outliers["cluster"] == cl_id
        cluster_emb = emb_outliers[mask]

        # Compute the centroid of the cluster
        centroid = cluster_emb.mean(axis=0, keepdims=True)  # shape (1, embedding_dim)

        # Compute cosine distances to known class centroids
        distances = cosine_distances(centroid, known_centroids)[
            0
        ]  # shape (num_classes,)

        # Find the minimum distance
        min_distance = np.min(distances)
        min_distance_class = known_class_info.index[np.argmin(distances)]
        distance_threshold = known_class_info.loc[min_distance_class, "mean_distance"]

        logger.info(
            f"Cluster {cl_id}: Min distance = {min_distance:.4f} to class {min_distance_class} and threshold = {distance_threshold:.4f}"
        )

        # If the minimum distance exceeds the threshold, label the cluster as a new class
        if min_distance > distance_threshold:
            df_outliers.loc[mask, "new_class"] = True
            logger.info(f"Cluster {cl_id} is a new class")
        else:
            logger.info(f"Cluster {cl_id} is an existing class")

    return df_outliers


def ensure_output_folder(output_folder):
    """Ensure the output folder exists."""
    os.makedirs(output_folder, exist_ok=True)


def save_plot(plot_path):
    """Save the current matplotlib figure to the specified path."""
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def plot_clusters(
    embeddings,
    labels,
    centroids=None,
    output_folder=".",
    filename="cluster_visualization.png",
):
    """Visualize clusters in 2D using PCA and save the plot."""
    ensure_output_folder(output_folder)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=labels,
        palette="viridis",
        alpha=0.8,
        edgecolor="k",
        s=50,
    )
    if centroids is not None:
        reduced_centroids = pca.transform(centroids)
        plt.scatter(
            reduced_centroids[:, 0],
            reduced_centroids[:, 1],
            c="red",
            marker="X",
            s=200,
            label="Centroids",
        )
    plt.title("Cluster Visualization in Reduced Dimensions", fontsize=16)
    plt.xlabel("PCA Component 1", fontsize=14)
    plt.ylabel("PCA Component 2", fontsize=14)
    plt.legend(title="Clusters", fontsize=12)
    plot_path = os.path.join(output_folder, filename)
    save_plot(plot_path)
    return plot_path


def silhouette_analysis(
    embeddings, labels, output_folder=".", filename="silhouette_analysis.png"
):
    """Perform and save a silhouette analysis plot."""
    ensure_output_folder(output_folder)

    score = silhouette_score(embeddings, labels)
    sample_silhouette_values = silhouette_samples(embeddings, labels)

    plt.figure(figsize=(12, 8))

    y_lower = 10
    for i in np.unique(labels):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            alpha=0.7,
            label=f"Cluster {i}",
        )

        y_lower = y_upper + 10

    plt.axvline(
        x=score,
        color="red",
        linestyle="--",
        label=f"Average Silhouette Score: {score:.2f}",
    )
    plt.title("Silhouette Analysis for Clustering", fontsize=16)
    plt.xlabel("Silhouette Coefficient Values", fontsize=14)
    plt.ylabel("Samples", fontsize=14)
    plt.legend(fontsize=12)
    plot_path = os.path.join(output_folder, filename)
    save_plot(plot_path)
    return plot_path


def plot_cluster_sizes(labels, output_folder=".", filename="cluster_sizes.png"):
    """Plot and save a bar chart of cluster sizes."""
    ensure_output_folder(output_folder)

    cluster_sizes = pd.Series(labels).value_counts()
    plt.figure(figsize=(12, 8))
    cluster_sizes.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title("Cluster Sizes Distribution", fontsize=16)
    plt.xlabel("Cluster ID", fontsize=14)
    plt.ylabel("Number of Samples", fontsize=14)

    plot_path = os.path.join(output_folder, filename)
    save_plot(plot_path)
    return plot_path


def generate_distance_to_centroids_report(
    outliers, output_folder=".", filename="distance_to_centroids.csv"
):
    """Save a CSV report of distances to centroids."""
    ensure_output_folder(output_folder)

    report_path = os.path.join(output_folder, filename)
    outliers[["cluster", "distance_to_centroids"]].to_csv(report_path, index=False)
    return report_path


def generate_cluster_summary_report(
    outliers_info, output_folder=".", filename="cluster_summary.csv"
):
    """Save a summary report of clusters."""
    ensure_output_folder(output_folder)

    report_path = os.path.join(output_folder, filename)
    outliers_info.to_csv(report_path, index=False)
    return report_path


def generate_heatmap_of_distances(
    outliers_centroids,
    known_centroids,
    output_folder=".",
    filename="distance_heatmap.png",
    outlier_labels=None,
    known_labels=None,
):
    """
    Generate and save a heatmap of distances between outlier and known centroids.

    Parameters:
        outliers_centroids (array-like): Centroids of outliers.
        known_centroids (array-like): Centroids of known classes.
        output_folder (str): Path to save the heatmap.
        filename (str): Name of the heatmap file.
        outlier_labels (list): Labels for outlier centroids (e.g., cluster IDs).
        known_labels (list): Labels for known centroids (e.g., class IDs).
    """
    ensure_output_folder(output_folder)

    # Compute pairwise distances
    distances = np.linalg.norm(
        np.expand_dims(np.vstack(outliers_centroids.values), axis=1)
        - np.vstack(known_centroids.values),
        axis=2,
    )

    # Generate and save heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        distances,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Distance"},
        yticklabels=outliers_centroids.index,  # Use provided outlier labels
        xticklabels=known_centroids.index,  # Use provided known labels
    )
    plt.title("Heatmap of Distances Between Outlier and Known Centroids", fontsize=16)
    plt.xlabel("Known Class Centroids", fontsize=14)
    plt.ylabel("Outlier Cluster Centroids", fontsize=14)

    plot_path = os.path.join(output_folder, filename)
    save_plot(plot_path)
    return plot_path


def save_all_reports_and_plots(
    embeddings, labels, centroids, outliers, outliers_info, output_folder
):
    """Generate and save all relevant reports and plots."""
    paths = {}

    # Save visualizations
    paths["cluster_visualization"] = plot_clusters(
        embeddings, labels, np.vstack(centroids.values), output_folder
    )
    paths["silhouette_analysis"] = silhouette_analysis(
        embeddings, labels, output_folder
    )
    paths["cluster_sizes"] = plot_cluster_sizes(labels, output_folder)

    # Save reports
    paths["distance_to_centroids"] = generate_distance_to_centroids_report(
        outliers, output_folder
    )
    paths["cluster_summary"] = generate_cluster_summary_report(
        outliers_info, output_folder
    )

    # Save heatmap
    paths["distance_heatmap"] = generate_heatmap_of_distances(
        outliers_centroids=outliers_info["centroid"],
        known_centroids=centroids,
        output_folder=output_folder,
        outlier_labels=outliers["cluster"].unique(),
    )

    return paths
