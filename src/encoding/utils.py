import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import mlflow
from tqdm import tqdm


def calculate_mean_cosine_score(vector, vector_error):
    # Ensure both inputs are NumPy arrays
    vector = np.array(vector)
    vector_error = np.array(vector_error)

    if vector.size == 0 or vector_error.size == 0:
        return np.nan  # Return NaN if there's no vector to compare

    return cosine_similarity(vector.reshape(1, -1), vector_error.reshape(1, -1))[0][0]


def calculate_most_likely_error(
    dataset: pd.DataFrame,
    errors: pd.DataFrame,
    dataset_vector_name: str,
    error_vector_name: str,
) -> pd.DataFrame:

    # Calculate the cosine similarity between the text_to_analyse and the errors
    for index, row in tqdm(
        errors.iterrows(), total=errors.shape[0], desc="Calculating cosine similarity"
    ):
        # Create a condition for filtering
        condition = dataset["CAR3"] == row["CODCAR3"]
        if row["CODCAR2"] != "0":
            condition &= dataset["CAR2"] == row["CODCAR2"]

        if not dataset.loc[condition, dataset_vector_name].empty:
            dataset.loc[condition, f'cosine_similarity_{row["ID_ERROR"]}'] = (
                dataset.loc[condition, dataset_vector_name].apply(
                    lambda x: calculate_mean_cosine_score(x, row[error_vector_name])
                )
            )

    # Get the most likely error
    # Get the columns with the cosine similarity
    cosine_columns = [col for col in dataset.columns if "cosine_similarity_" in col]

    dataset[cosine_columns] = dataset[cosine_columns].fillna(0)  # Fill NA with 0

    # Get the highest score and the error with the highest score
    dataset.loc[:, "highest_score"] = dataset[cosine_columns].max(axis=1)
    dataset.loc[:, "highest_score_error"] = (
        dataset[cosine_columns].idxmax(axis=1).apply(lambda x: x.split("_")[-1])
    )

    return dataset


def custom_grid_search(
    pipeline: Pipeline,
    parameters: dict,
    X: Any,
    scoring: Any = None,
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
    random_state: int = 42,
):

    grid_search = GridSearchCV(
        pipeline, parameters, cv=cv, n_jobs=n_jobs, verbose=verbose
    )

    grid_search.fit(X)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_

    return best_params, best_score, best_estimator


def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    score = silhouette_score(X, labels)
    return score


def generate_unsupervised_report(
    env: str,
    data: pd.DataFrame,
    model_name: str,
    dim_reduction_model: list[str],
    clustering: bool = False,
    most_common_words: bool = False,
) -> None:
    """
    Generate an unsupervised report.
    :param env: Environment
    :param data: Data
    :param model_name: Model name [TF-IDF, Word2Vec, SentenceTransformer]
    :param dim_reduction_model: Dimensionality reduction model [PCA, TSNE]
    :param clustering: Clustering flag
    :param most_common_words: Most common words flag
    :return: None
    """

    report_folder_path = f"reports/{env}/{model_name}"
    os.makedirs(report_folder_path, exist_ok=True)

    if most_common_words:
        raise NotImplementedError("Most common words not implemented yet")

    if clustering:
        vector_df = data.drop(columns=["cluster", "codigo", "id_pieza"])
        if "PCA" in dim_reduction_model:
            save_path = os.path.join(
                report_folder_path, f"{model_name}_clustering_PCA.png"
            )
            pca = PCA(n_components=3)
            pca_vector = pca.fit_transform(vector_df)
            pca_vector = pd.DataFrame(pca_vector, columns=["PC1", "PC2", "PC3"])

            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                pca_vector["PC1"],
                pca_vector["PC2"],
                pca_vector["PC3"],
                c=data["cluster"],
                cmap="viridis",
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            plt.title(
                f"Clustering of the text data using {model_name} for "
                f"encoding and PCA for dimensionality reduction"
            )

            # Save the plot to the specified path
            plt.savefig(save_path)
            plt.close()  # Close the figure to free memory

        if "TSNE" in dim_reduction_model:
            save_path = os.path.join(
                report_folder_path, f"{model_name}_clustering_TSNE.png"
            )

            tsne = TSNE(n_components=3)
            tsne_vector = tsne.fit_transform(vector_df)
            tsne_vector = pd.DataFrame(tsne_vector, columns=["TSNE1", "TSNE2", "TSNE3"])

            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                tsne_vector["TSNE1"],
                tsne_vector["TSNE2"],
                tsne_vector["TSNE3"],
                c=data["cluster"],
                cmap="viridis",
            )
            ax.set_xlabel("TSNE1")
            ax.set_ylabel("TSNE2")
            ax.set_zlabel("TSNE3")
            plt.title(
                f"Clustering of the text data using {model_name} for "
                f"encoding and TSNE for dimensionality reduction"
            )

            # Save the plot to the specified path
            plt.savefig(save_path)
            plt.close()  # Close the figure to free memory
