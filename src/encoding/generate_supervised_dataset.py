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
) -> None:
    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start generating supervised dataset in {env} environment")

    # Load reviewed data
    logger.info(f"Load reviewed data")
    reviewed_data = pd.read_parquet(input_reviewed_data)

    logger.info(f"Save supervised dataset")
    reviewed_data.to_parquet(output_supervised_dataset)


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
    args = parser.parse_args()

    generate_supevised_dataset(
        env=args.env,
        input_reviewed_data=args.input_reviewed_data,
        output_supervised_dataset=args.output_supervised_dataset,
    )
