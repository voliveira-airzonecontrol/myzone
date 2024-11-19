import argparse
import concurrent.futures
import csv
import os
import time
import multiprocessing
import concurrent.futures

import joblib
import pandas as pd

from src.utils import load_config, get_logger, load_data, save_data


def generate_unsupervised_dataset(
    env: str,
    input_tfidf_encoded_data: str,
    input_doc2vec_encoded_data: str,
    input_sentence_transformer_encoded_data: str,
    input_preprocessed_data: str,
    output_unsupervised_tfidf_dataset: str,
    output_unsupervised_doc2vec_dataset: str,
    output_unsupervised_sentence_transformer_dataset: str,
    input_tfidf_model: str,
    input_doc2vec_model: str,
) -> None:
    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start generating unsupervised dataset in {env} environment")

    # Load preprocessed data
    logger.info(f"Load preprocessed data")
    preprocessed_data = load_data(
        data_path=input_preprocessed_data,
        step_config=processing_config.processing.preprocessed_data,
    )

    # Load TF-IDF encoded data
    logger.info(f"Load TF-IDF encoded data")
    tfidf_encoded_data = load_data(
        data_path=input_tfidf_encoded_data,
        step_config=processing_config.processing.encoded_data,
    )

    # Load Doc2Vec encoded data
    logger.info(f"Load Doc2Vec encoded data")
    doc2vec_encoded_data = load_data(
        data_path=input_doc2vec_encoded_data,
        step_config=processing_config.processing.encoded_data,
    )

    # Load SentenceTransformer encoded data
    logger.info(f"Load SentenceTransformer encoded data")
    sentence_transformer_encoded_data = load_data(
        data_path=input_sentence_transformer_encoded_data,
        step_config=processing_config.processing.encoded_data,
    )

    # Load TF-IDF model with joblib
    logger.info(f"Load TF-IDF model")
    tfidf_model = joblib.load(input_tfidf_model)

    # Load Doc2Vec model with joblib
    logger.info(f"Load Doc2Vec model")
    doc2vec_model = joblib.load(input_doc2vec_model)

    # Generate unsupervised dataset




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to preprocess the data from",
        default="dev",
    )
    parser.add_argument(
        "--input-tfidf-encoded-data",
        type=str,
        help="Path to the input TF-IDF encoded data",
    )
    parser.add_argument(
        "--input-doc2vec-encoded-data",
        type=str,
        help="Path to the input Doc2Vec encoded data",
    )
    parser.add_argument(
        "--input-sentence-transformer-encoded-data",
        type=str,
        help="Path to the input SentenceTransformer encoded data",
    )
    parser.add_argument(
        "--input-preprocessed-data",
        type=str,
        help="Path to the input preprocessed data",
    )
    parser.add_argument(
        "--output-unsupervised-tfidf-dataset",
        type=str,
        help="Path to save the output TF-IDF unsupervised dataset",
    )
    parser.add_argument(
        "--output-unsupervised-doc2vec-dataset",
        type=str,
        help="Path to save the output Doc2Vec unsupervised dataset",
    )
    parser.add_argument(
        "--output-unsupervised-sentence-transformer-dataset",
        type=str,
        help="Path to save the output SentenceTransformer unsupervised dataset",
    )
    parser.add_argument(
        "--input-tfidf-model",
        type=str,
        help="Path to the input TF-IDF model",
    )
    parser.add_argument(
        "--input-doc2vec-model",
        type=str,
        help="Path to the input Doc2Vec model",
    )

    args = parser.parse_args()

    generate_unsupervised_dataset(
        env=args.env,
        input_tfidf_encoded_data=args.input_tfidf_encoded_data,
        input_doc2vec_encoded_data=args.input_doc2vec_encoded_data,
        input_sentence_transformer_encoded_data=args.input_sentence_transformer_encoded_data,
        input_preprocessed_data=args.input_preprocessed_data,
        output_unsupervised_tfidf_dataset=args.output_unsupervised_tfidf_dataset,
        output_unsupervised_doc2vec_dataset=args.output_unsupervised_doc2vec_dataset,
        output_unsupervised_sentence_transformer_dataset=args.output_unsupervised_sentence_transformer_dataset,
        input_tfidf_model=args.input_tfidf_model,
        input_doc2vec_model=args.input_doc2vec_model,
    )
