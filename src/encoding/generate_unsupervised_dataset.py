import argparse
import concurrent.futures
import csv
import os
import time
import multiprocessing
import concurrent.futures

import pandas as pd

from src.utils import load_config, get_logger, load_data, save_data


def generate_unsupervised_dataset(
        env: str,
) -> None:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to preprocess the data from",
        default="dev",
    )

    args = parser.parse_args()

    generate_unsupervised_dataset(
        env=args.env,
    )
