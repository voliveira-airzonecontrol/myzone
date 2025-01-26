import argparse
import os

from src.clustering.utils import (
    find_centroids,
    find_distance_to_centroids,
    perform_dbscan_grid_search,
    cluster_and_find_new_class,
    save_all_reports_and_plots,
    ensure_output_folder,
)
from src.training.model import TransformerClassifier
from src.utils import load_config, get_logger

import pandas as pd
import numpy as np


def cluster_outliers(
    env: str,
    input_dataset: str,
    input_outliers: str,
    input_model: str,
    output_new_class: str,
    output_reports: str,
    model_type: str = "BERT",
) -> None:
    """
    Cluster the outliers using DBSCAN, find new classes and save the results.
    """
    config = load_config(file_name="config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)
    logger.info("--------------------------------------------------")
    logger.info(f"Running outlier detection in {env} environment.")

    # Load dataset
    logger.info("Loading datasets...")
    known_classes = pd.read_parquet(input_dataset)
    outliers_found = pd.read_parquet(input_outliers)

    # Get the features and target names
    features = training_config.training[model_type].features
    target = training_config.training[model_type].target

    # Filter the features and target columns
    known_classes = known_classes[[features, target]]
    outliers_found = outliers_found[[features, target]]

    # # Get the number of unique labels
    num_labels = known_classes[target].nunique()

    # Load the model
    logger.info("Loading the model...")
    clf = TransformerClassifier(
        model_name=None,
        num_labels=num_labels,
        local_model_path=input_model,
    )

    # ---------------------------------------------------------------------
    # Get embeddings for known classes and outliers
    # ---------------------------------------------------------------------
    logger.info("Getting embeddings for known classes...")
    # Generate embeddings for known classes
    embeddings = clf.get_embeddings(known_classes[features].values)
    known_classes["embeddings"] = [e for e in embeddings]

    logger.info("Getting embeddings for outliers...")
    # Generate embeddings for outliers
    embeddings = clf.get_embeddings(outliers_found[features].values)
    outliers_found["embeddings"] = [e for e in embeddings]

    # ---------------------------------------------------------------------
    # Clustering of outliers
    # ---------------------------------------------------------------------
    logger.info("Clustering outliers...")
    # Format embeddings of outliers
    embeddings = np.vstack(outliers_found["embeddings"])

    # Define a parameter grid for DBSCAN
    param_grid = {
        "eps": np.arange(0.1, 30, 0.1),
        "min_samples": range(10, 20, 1),
        "metric": ["cosine", "euclidean"],
    }

    # Perform grid search
    results, best_model = perform_dbscan_grid_search(embeddings, param_grid)

    # Display results
    logger.info(f"Best Parameters: {results['best_params']}")
    logger.info(f"Best Silhouette Score: {results['best_score']}")

    # Retrieve cluster labels from the best model
    cluster_labels = best_model.labels_
    logger.info(f"Cluster Labels: {cluster_labels}")

    outliers_found["cluster"] = cluster_labels

    # Best metric for distance calculation
    best_metric = results["best_params"]["metric"]

    # ---------------------------------------------------------------------
    # Find centroids for known classes and outliers
    # ---------------------------------------------------------------------
    logger.info("Finding centroids for known classes...")
    centroids = find_centroids(known_classes, target)

    # ---------------------------------------------------------------------
    # Find mean distance of each row to the centroids of known classes
    # ---------------------------------------------------------------------
    logger.info(
        "Finding mean distance of each row to the centroids of known classes..."
    )
    known_classes["distance_to_centroids"] = find_distance_to_centroids(
        known_classes, centroids, distance_type=best_metric
    )
    # Calculate mean distance from centroids of each known classes
    mean_distance = known_classes.groupby(target)["distance_to_centroids"].mean()
    # Calculate standard deviation of mean distance from centroids of each known classes
    std_distance = known_classes.groupby(target)["distance_to_centroids"].std()
    # Join the centroids and mean distance of known classes
    known_classes_info = centroids.to_frame("centroid").join(
        mean_distance.to_frame("mean_distance")
    )
    # Join the standard deviation of mean distance of known classes
    known_classes_info = known_classes_info.join(std_distance.to_frame("std_distance"))
    # Calculate threshold for each known classes
    known_classes_info["threshold"] = known_classes_info[
        "mean_distance"
    ] + known_classes_info["std_distance"]

    # ---------------------------------------------------------------------
    # Find new classes
    # ---------------------------------------------------------------------
    outliers_centroids = find_centroids(outliers_found, "cluster")
    outliers_found["distance_to_centroids"] = find_distance_to_centroids(
        outliers_found, outliers_centroids, target_column="cluster", distance_type=best_metric
    )
    outliers_info = outliers_centroids.to_frame("centroid").join(
        outliers_found.groupby("cluster")["distance_to_centroids"]
        .mean()
        .to_frame("mean_distance")
    )

    new_class = cluster_and_find_new_class(
        df_outliers=outliers_found,
        known_class_info=known_classes_info,
        logger=logger,
        distance_type=best_metric,
    )
    logger.info(
        f"Total new classes found:  {len(new_class[new_class['new_class']]['cluster'].unique())}"
    )

    # ---------------------------------------------------------------------
    # Save the new class
    # ---------------------------------------------------------------------
    ensure_output_folder(output_reports)
    new_class.to_parquet(output_new_class)

    # ---------------------------------------------------------------------
    # Generate all reports and plots
    # ---------------------------------------------------------------------
    logger.info("Saving reports and plots...")
    paths = save_all_reports_and_plots(
        embeddings=embeddings,
        labels=cluster_labels,
        centroids=outliers_centroids,
        outliers=outliers_found,
        outliers_info=outliers_info,
        output_folder=output_reports,
    )
    logger.info(f"Reports and plots saved in: {paths}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Outlier detection using MAPIE conformal sets."
    )
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev, prod, etc.)"
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        help="Path to the input dataset",
        required=True,
    )
    parser.add_argument(
        "--input-outliers",
        type=str,
        help="Path to the input outliers",
        required=True,
    )
    parser.add_argument(
        "--input-model",
        type=str,
        help="Path to the input model",
        required=True,
    )
    parser.add_argument(
        "--output-new-class",
        type=str,
        help="Path to the output new classes",
        required=True,
    )
    parser.add_argument(
        "--output-reports",
        type=str,
        help="Path to the output reports",
        required=True,
    )
    args = parser.parse_args()

    cluster_outliers(
        env=args.env,
        input_dataset=args.input_dataset,
        input_outliers=args.input_outliers,
        input_model=args.input_model,
        output_new_class=args.output_new_class,
        output_reports=args.output_reports,
    )
