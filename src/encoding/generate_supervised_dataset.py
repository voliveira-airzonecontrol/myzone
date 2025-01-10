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


def generate_supevised_dataset(
    env: str,
    input_reviewed_data: str,
    output_supervised_dataset: str,
    output_supervised_dataset_phase2: str,
) -> None:
    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start generating supervised dataset in {env} environment")

    # Load reviewed data
    logger.info(f"Load reviewed data")
    reviewed_data = pd.read_csv(input_reviewed_data)
    df = reviewed_data[
        [
            "codigo",
            "id_pieza",
            "CODART_A3",
            "CAR3",
            "text_to_analyse",
            "processed_text_to_analyse",
            "Cod_error",
            "highest_score_error",
        ]
    ]
    clean_df = df[df["CAR3"] == 91]
    clean_df = clean_df[
        (~clean_df["Cod_error"].isna())
        & (~clean_df["Cod_error"].isin(["na", "n", "nn", "309", " na"]))
    ]
    clean_df = clean_df.rename(
        columns={"Cod_error": "label", "highest_score_error": "similarity_prediction"}
    )
    clean_df["Ground_truth"] = clean_df["label"]
    clean_df["label"] = clean_df["label"].astype(int)

    # Standardize the labels
    clean_df["label"] = clean_df["label"].replace(
        {301: 0, 302: 2, 303: 1, 308: 3, 304: 4, 305: 5, 309: 6}
    )

    logger.info(f"Save supervised dataset")
    clean_df[clean_df["label"].isin([0, 1])].to_parquet(output_supervised_dataset)
    clean_df[clean_df["label"] == 2].to_parquet(output_supervised_dataset_phase2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to preprocess the data from",
        default="dev",
    )
    parser.add_argument(
        "--input-reviewed-data",
        type=str,
        help="Path to the reviewed data",
    )
    parser.add_argument(
        "--output-supervised-dataset",
        type=str,
        help="Path to save the supervised dataset",
    )
    parser.add_argument(
        "--output-supervised-dataset-phase2",
        type=str,
        help="Path to save the supervised dataset phase 2",
    )
    args = parser.parse_args()

    generate_supevised_dataset(
        env=args.env,
        input_reviewed_data=args.input_reviewed_data,
        output_supervised_dataset=args.output_supervised_dataset,
        output_supervised_dataset_phase2=args.output_supervised_dataset_phase2,
    )
