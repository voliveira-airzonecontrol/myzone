import os
from typing import Any, Optional

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import mlflow


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


def log_to_mlflow(
    run_name: str,
    params: dict,
    metrics: dict,
    model_name: str,
    model: Any,
    model_type: str,
    artifacts: Optional[dict] = None,
    experiment_name: str = "Default Experiment",
    tracking_uri: str = "http://192.168.2.241:5000",
    nested: bool = False,
) -> None:
    """
    Log the model to MLflow.
    :param model_type: Model type (sklearn, keras, pytorch, xgboost)
    :param run_name: Run name
    :param model_name: Model name
    :param params: Parameters
    :param metrics: Metrics
    :param model: Model
    :param experiment_name:
    :param tracking_uri:
    :param nested:
    :return: None
    """

    # Set the tracking URI to the specified MLflow server
    mlflow.set_tracking_uri(tracking_uri)

    # Create or set the experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if model:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path=model_type)
            elif model_type == "keras":
                mlflow.keras.log_model(model, artifact_path=model_type)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, artifact_path=model_type)
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path=model_type)

            # Register the model in the Model Registry with a specific name
            model_uri = f"runs:/{run.info.run_id}/{model_type}"
            model_version = mlflow.register_model(model_uri, name=model_name)

            print(
                f"Model {model_version.name} registered with version: {model_version.version}"
            )

        if artifacts:
            for key, value in artifacts.items():
                mlflow.log_artifact(value, key)


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
