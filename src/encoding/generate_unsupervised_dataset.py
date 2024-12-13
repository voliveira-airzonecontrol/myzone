import argparse
import concurrent.futures
import csv
import os
import time
import multiprocessing
import concurrent.futures

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.encoding.utils import calculate_mean_cosine_score, calculate_most_likely_error
from src.preprocessing.utils import pre_process_text_spacy
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
    input_tipo_error: str,
) -> None:
    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    training_config = load_config(file_name="training_config", env=env)
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

    # Load models with joblib
    logger.info(f"Load models")
    tfidf_model = joblib.load(input_tfidf_model)
    doc2vec_model = joblib.load(input_doc2vec_model)
    sentence_transformer_model = SentenceTransformer(
        training_config.training.sentence_transformer.transformer_name
    )

    # Read list of errors
    logger.info(f"Read and process list of errors")
    errors = load_data(
        data_path=input_tipo_error, step_config=processing_config.processing.tipo_error
    )
    errors["DESCRIPCION"] = (
        errors["MOTIVO"] + " " + errors["DESCRIPCION"]
    )  # Concatenate MOTIVO and DESCRIPCION
    errors = errors.fillna("")

    errors["description_processed"] = pre_process_text_spacy(
        errors["DESCRIPCION"].values,
        **processing_config.processing["tokenizer"],
    )

    # Infer vectors for each model for each error
    logger.info(f"Infer vectors for each model for each error")
    errors["vector_tfidf"] = list(
        tfidf_model.transform(errors["description_processed"]).toarray()
    )
    errors["vector_doc2vec"] = list(
        doc2vec_model.transform(errors["description_processed"])
    )
    errors["vector_sentence_transformer"] = list(
        sentence_transformer_model.encode(errors["description_processed"].values)
    )

    # Get the vectors for the preprocessed data for tfidf
    vector_tfidf = (
        tfidf_encoded_data.drop(columns=["codigo", "id_pieza", "cluster"])
        .to_numpy()
        .tolist()
    )
    df_tfidf = pd.DataFrame({"vector_tfidf": vector_tfidf})

    # Get the vectors for the preprocessed data for doc2vec
    vector_doc2vec = (
        doc2vec_encoded_data.drop(columns=["codigo", "id_pieza", "cluster"])
        .to_numpy()
        .tolist()
    )
    df_doc2vec = pd.DataFrame({"vector_doc2vec": vector_doc2vec})

    # Get the vectors for the preprocessed data for sentence transformer
    vector_sentence_transformer = (
        sentence_transformer_encoded_data.drop(
            columns=["codigo", "id_pieza", "cluster"]
        )
        .to_numpy()
        .tolist()
    )
    df_sentence_transformer = pd.DataFrame(
        {"vector_sentence_transformer": vector_sentence_transformer}
    )

    logger.info(
        "Calculate most likely error for tfidf encoded data"
    )  # ---------------------------------------------
    # Concatenate the vectors with the original data
    unsupervised_tfidf_dataset = pd.concat(
        objs=[
            preprocessed_data.reset_index(drop=True),
            df_tfidf.reset_index(drop=True),
        ],
        axis=1,
    )
    # Calculate the mean cosine score for each error and the most likely error
    unsupervised_tfidf_dataset = calculate_most_likely_error(
        dataset=unsupervised_tfidf_dataset,
        errors=errors,
        dataset_vector_name="vector_tfidf",
        error_vector_name="vector_tfidf",
    )
    # Save the tfidf unsupervised dataset
    save_data(
        data=unsupervised_tfidf_dataset,
        output_path=output_unsupervised_tfidf_dataset,
    )

    logger.info(
        "Calculate most likely error for doc2vec encoded data"
    )  # ------------------------------------------
    # Concatenate the vectors with the original data
    unsupervised_doc2vec_dataset = pd.concat(
        objs=[
            preprocessed_data.reset_index(drop=True),
            df_doc2vec.reset_index(drop=True),
        ],
        axis=1,
    )
    # Calculate the mean cosine score for each error and the most likely error
    unsupervised_doc2vec_dataset = calculate_most_likely_error(
        dataset=unsupervised_doc2vec_dataset,
        errors=errors,
        dataset_vector_name="vector_doc2vec",
        error_vector_name="vector_doc2vec",
    )
    # Save the doc2vec unsupervised dataset
    save_data(
        data=unsupervised_doc2vec_dataset,
        output_path=output_unsupervised_doc2vec_dataset,
    )

    logger.info(
        "Calculate most likely error for sentence transformer encoded data"
    )  # ------------------------------
    # Concatenate the vectors with the original data
    unsupervised_sentence_transformer_dataset = pd.concat(
        objs=[
            preprocessed_data.reset_index(drop=True),
            df_sentence_transformer.reset_index(drop=True),
        ],
        axis=1,
    )
    # Calculate the mean cosine score for each error and the most likely error
    unsupervised_sentence_transformer_dataset = calculate_most_likely_error(
        dataset=unsupervised_sentence_transformer_dataset,
        errors=errors,
        dataset_vector_name="vector_sentence_transformer",
        error_vector_name="vector_sentence_transformer",
    )
    # Save the sentence transformer unsupervised dataset
    save_data(
        data=unsupervised_sentence_transformer_dataset,
        output_path=output_unsupervised_sentence_transformer_dataset,
    )


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
    parser.add_argument(
        "--input-tipo-error",
        type=str,
        help="Path to the input Tipo Error data",
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
        input_tipo_error=args.input_tipo_error,
    )
