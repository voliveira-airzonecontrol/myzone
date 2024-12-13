import argparse
import concurrent.futures
import csv
import os
import time
import multiprocessing
import concurrent.futures

import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from src.encoding.utils import (
    custom_grid_search,
    log_to_mlflow,
    generate_unsupervised_report,
)
from src.utils import load_config, get_logger, load_data, save_data


def sentence_transformer_encoding(
    env: str,
    input_data: str,
    output_sentence_transformer_encoded_data: str,
    output_sentence_transformer_model: str,
) -> None:
    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)

    mlflow.set_tracking_uri(training_config.mlflow.tracking_uri)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start SentenceTransformer encoding in {env} environment")

    # Load preprocessed data
    logger.info(f"Load preprocessed data")
    preprocessed_data = load_data(
        data_path=input_data,
        step_config=processing_config.processing.preprocessed_data,
    )

    X = preprocessed_data["processed_text_to_analyse"].values

    # Load model
    model = SentenceTransformer(
        training_config.training.sentence_transformer.transformer_name
    )

    logger.info(f"Encoding data")
    X_embeddings = model.encode(X)

    # Create a pipeline to perform the TF-IDF encoding
    pipeline = Pipeline(
        [
            (
                training_config.training.sentence_transformer.model,
                KMeans(random_state=training_config.training.random_state),
            ),
        ]
    )

    parameters = {
        f"{training_config.training.sentence_transformer.model}__n_clusters": training_config.training.sentence_transformer.n_clusters,
    }

    # Fit the pipeline
    logger.info("Fit the pipeline")
    best_params, best_score, best_estimator = custom_grid_search(
        pipeline=pipeline, parameters=parameters, X=X_embeddings
    )

    log_to_mlflow(
        run_name="SentenceTransformer encoding",
        params=best_params,
        metrics={"mean_test_score": best_score},
        model_name="SentenceTransformer encoding",
        model=None,
        artifacts=None,
        model_type="sklearn",
        experiment_name="SentenceTransformer encoding",
        tracking_uri=training_config.mlflow.tracking_uri,
    )

    # Transform the data
    logger.info("Transform the data")
    vector_df = pd.DataFrame(
        X_embeddings, columns=[f"vector_{i}" for i in range(X_embeddings.shape[1])]
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
        training_config.training.sentence_transformer.model
    ].predict(X_embeddings)

    logger.info(f"Generating results report")
    # Generate a report
    generate_unsupervised_report(
        env=env,
        data=preprocessed_data,
        model_name="SentenceTransformer",
        dim_reduction_model=["PCA", "TSNE"],
        clustering=True,
        most_common_words=False,
    )

    # Save the data
    logger.info(f"Save the data")
    save_data(
        data=preprocessed_data,
        output_path=output_sentence_transformer_encoded_data,
    )

    """# Save model
    logger.info(f"Save the model")
    os.makedirs(os.path.dirname(output_sentence_transformer_model), exist_ok=True)
    model.save(output_sentence_transformer_model)
"""


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
        help="Path to the input preprocessed data",
    )
    parser.add_argument(
        "--output-sentence-transformer-encoded-data",
        type=str,
        help="Path to save the output corpus",
    )
    parser.add_argument(
        "--output_sentence_transformer_model",
        type=str,
        help="Path to save the output model",
    )

    args = parser.parse_args()

    sentence_transformer_encoding(
        env=args.env,
        input_data=args.input_data,
        output_sentence_transformer_encoded_data=args.output_sentence_transformer_encoded_data,
        output_sentence_transformer_model=args.output_sentence_transformer_model,
    )
