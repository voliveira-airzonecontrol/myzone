import argparse
import concurrent.futures
import csv
import os
import time
import multiprocessing
import concurrent.futures
import string
from joblib import dump

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import silhouette_score
import mlflow

from src.encoding.encoders import TfIdfPreprocessor
from src.encoding.utils import (
    custom_grid_search,
    silhouette_scorer,
    log_to_mlflow,
    generate_unsupervised_report,
)
from src.utils import load_config, get_logger, load_data, save_data


def tfidf_encoding(
    env: str,
    input_data: str,
    output_tfidf_encoded_data: str,
    output_tfidf_model: str,
) -> None:
    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)

    mlflow.set_tracking_uri(training_config.mlflow.tracking_uri)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start TF-IDF encoding in {env} environment")

    # Load preprocessed data
    logger.info(f"Load corpus data")
    preprocessed_data = load_data(
        data_path=input_data,
        step_config=processing_config.processing.preprocessed_data,
    )

    X = preprocessed_data["processed_text_to_analyse"].values

    # Create a pipeline to perform the TF-IDF encoding
    pipeline = Pipeline(
        [
            (training_config.training.tfidf.processor, TfIdfPreprocessor()),
            (
                training_config.training.tfidf.model,
                KMeans(random_state=training_config.training.random_state),
            ),
        ]
    )

    parameters = {
        f"{training_config.training.tfidf.processor}__min_df": training_config.training.tfidf.min_df,
        f"{training_config.training.tfidf.processor}__max_df": training_config.training.tfidf.max_df,
        f"{training_config.training.tfidf.model}__n_clusters": training_config.training.tfidf.n_clusters,
    }

    # Fit the pipeline
    logger.info("Fit the pipeline")
    best_params, best_score, best_estimator = custom_grid_search(
        pipeline=pipeline, parameters=parameters, X=X
    )

    log_to_mlflow(
        run_name="TF-IDF encoding",
        params=best_params,
        metrics={"mean_test_score": best_score},
        model_name="TF-IDF encoding",
        model=None,
        artifacts=None,
        model_type="sklearn",
        experiment_name="TF-IDF encoding",
        tracking_uri=training_config.mlflow.tracking_uri,
    )

    # Transform the data
    logger.info("Transform the data")
    # Add the cluster to the dataset
    vector = best_estimator.named_steps[
        training_config.training.tfidf.processor
    ].transform(X)
    vector_df = pd.DataFrame(
        vector.toarray(), columns=[f"vector_{i}" for i in range(vector.shape[1])]
    )

    # Concatenate the vectors with the original data
    preprocessed_data = pd.concat(
        objs=[
            preprocessed_data[["codigo", "id_pieza"]].reset_index(drop=True),
            vector_df.reset_index(drop=True),
        ],
        axis=1,
    )

    preprocessed_data["cluster"] = best_estimator.named_steps[
        training_config.training.tfidf.model
    ].predict(vector)

    logger.info(f"Generating results report")
    # Generate a report
    generate_unsupervised_report(
        env=env,
        data=preprocessed_data,
        model_name="TF-IDF",
        dim_reduction_model=["PCA", "TSNE"],
        clustering=True,
        most_common_words=False,
    )

    # Save the data
    logger.info(f"Save the data")
    save_data(
        data=preprocessed_data,
        output_path=output_tfidf_encoded_data,
    )

    # Save the KMeans model
    logger.info(f"Save the model")
    os.makedirs(os.path.dirname(output_tfidf_model), exist_ok=True)
    model = best_estimator.named_steps[training_config.training.tfidf.processor]
    dump(value=model, filename=output_tfidf_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to preprocess the data from",
        default="dev",
    )
    parser.add_argument(
        "--input-data",
        type=str,
        help="Path to the input corpus",
    )
    parser.add_argument(
        "--output-tfidf-encoded-data",
        type=str,
        help="Path to save the output corpus",
    )
    parser.add_argument(
        "--output-tfidf-model",
        type=str,
        help="Path to save the output model",
    )

    args = parser.parse_args()

    tfidf_encoding(
        env=args.env,
        input_data=args.input_data,
        output_tfidf_encoded_data=args.output_tfidf_encoded_data,
        output_tfidf_model=args.output_tfidf_model,
    )
