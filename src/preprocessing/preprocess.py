import argparse
import concurrent.futures
import csv
import os
import time
import multiprocessing

import pandas as pd
import spacy
import gensim.models.doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

from src.utils import load_config, get_logger, load_data

nlp = spacy.load("es_core_news_sm")


def preprocess_text(docs):
    """
    Function to preprocess the text
    """
    # Ensure all entries are strings
    docs = docs.fillna("").astype(str)
    # Process the text
    texts = [doc for doc in nlp.pipe(docs, disable=["ner", "parser"])]
    processed_texts = []
    for doc in texts:
        tokens = [
            token.text.lower()
            for token in doc
            if not token.is_punct and not token.is_stop and not token.is_space
        ]
        processed_texts.append(" ".join(tokens))
    return processed_texts


def preprocess(
        env: str
) -> None:
    config = load_config(file_name="config", env=env)
    data_config = load_config(file_name="data_config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start preprocessing data in {env} environment")

    # Load data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to preprocess the data from",
        default="dev",
    )

    args = parser.parse_args()

    preprocess(args.env)
