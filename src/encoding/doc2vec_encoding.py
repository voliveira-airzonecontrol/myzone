import argparse
import concurrent.futures
import csv
import os
import time
import multiprocessing
import concurrent.futures
from joblib import dump

import pandas as pd
import mlflow
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from src.encoding.encoders import Doc2VecPreprocessor
from src.encoding.utils import (
    custom_grid_search,
    log_to_mlflow,
    generate_unsupervised_report,
)
from src.preprocessing.utils import pre_process_text_spacy
from src.utils import load_config, get_logger, load_data, save_data


def doc2vec_encoding(
    env: str,
    input_data: str,
    input_corpus: str,
    output_doc2vec_encoded_data: str,
    output_doc2vec_model: str,
) -> None:
    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)

    mlflow.set_tracking_uri(training_config.mlflow.tracking_uri)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start Doc2Vec encoding in {env} environment")

    # Load preprocessed data
    logger.info(f"Load corpus data")
    preprocessed_data = load_data(
        data_path=input_data,
        step_config=processing_config.processing.preprocessed_data,
    )
    corpus_data = load_data(
        data_path=input_corpus,
        step_config=processing_config.processing.corpus,
    )

    logger.info("Preprocess corpus data")
    corpus_data["processed_text"] = pre_process_text_spacy(
        corpus_data["text_to_analyse"].values,
        **processing_config.processing["tokenizer"],
    )

    corpus_data = corpus_data[
        (corpus_data["processed_text"].notnull())
        & (corpus_data["processed_text"] != "")
    ]

    X = corpus_data["processed_text"].values

    # Create a pipeline to perform the Doc2Vec encoding
    pipeline = Pipeline(
        [
            (training_config.training.doc2vec.processor, Doc2VecPreprocessor()),
            (
                training_config.training.doc2vec.model,
                KMeans(random_state=training_config.training.random_state),
            ),
        ]
    )

    parameters = {
        f"{training_config.training.doc2vec.processor}__dm": training_config.training.doc2vec.dm,
        f"{training_config.training.doc2vec.processor}__vector_size": training_config.training.doc2vec.vector_size,
        f"{training_config.training.doc2vec.processor}__epochs": training_config.training.doc2vec.epochs,
        f"{training_config.training.doc2vec.processor}__min_count": training_config.training.doc2vec.min_count,
        f"{training_config.training.doc2vec.processor}__sample": training_config.training.doc2vec.sample,
        f"{training_config.training.doc2vec.processor}__workers": [
            multiprocessing.cpu_count()
        ],
        f"{training_config.training.doc2vec.processor}__negative": training_config.training.doc2vec.negative,
        f"{training_config.training.doc2vec.processor}__seed": [
            training_config.training.random_state
        ],
        f"{training_config.training.doc2vec.processor}__hs": training_config.training.doc2vec.hs,
        f"{training_config.training.doc2vec.model}__n_clusters": training_config.training.doc2vec.n_clusters,
    }

    # Fit the pipeline
    logger.info("Fit the pipeline")
    best_params, best_score, best_estimator = custom_grid_search(
        pipeline=pipeline, parameters=parameters, X=X
    )

    log_to_mlflow(
        run_name="Doc2Vec encoding",
        params=best_params,
        metrics={"mean_test_score": best_score},
        model_name="Doc2Vec encoding",
        model=None,
        artifacts=None,
        model_type="gensim",
        experiment_name="Doc2Vec encoding",
        tracking_uri=training_config.mlflow.tracking_uri,
    )

    # Transform the data
    logger.info("Transform the data")
    # Add the cluster to the dataset
    vector = best_estimator.named_steps[
        training_config.training.doc2vec.processor
    ].transform(X)
    vector_df = pd.DataFrame(
        vector, columns=[f"vector_{i}" for i in range(vector.shape[1])]
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
        training_config.training.doc2vec.model
    ].predict(vector)

    logger.info(f"Generating results report")
    # Generate a report
    generate_unsupervised_report(
        env=env,
        data=preprocessed_data,
        model_name="Doc2Vec",
        dim_reduction_model=["PCA", "TSNE"],
        clustering=True,
        most_common_words=False,
    )

    # Save the data
    logger.info(f"Save the data")
    save_data(
        data=preprocessed_data,
        output_path=output_doc2vec_encoded_data,
    )

    # Save model
    logger.info(f"Save the model")
    os.makedirs(os.path.dirname(output_doc2vec_model), exist_ok=True)
    model = best_estimator.named_steps[training_config.training.doc2vec.processor]
    dump(model, output_doc2vec_model)


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
        "--input-corpus",
        type=str,
        help="Path to the input corpus",
    )
    parser.add_argument(
        "--output-doc2vec-encoded-data",
        type=str,
        help="Path to save the output corpus",
    )
    parser.add_argument(
        "--output-doc2vec-model",
        type=str,
        help="Path to save the output model",
    )

    args = parser.parse_args()

    doc2vec_encoding(
        env=args.env,
        input_data=args.input_data,
        input_corpus=args.input_corpus,
        output_doc2vec_encoded_data=args.output_doc2vec_encoded_data,
        output_doc2vec_model=args.output_doc2vec_model,
    )
