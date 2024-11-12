import argparse
import concurrent.futures
import csv
import os
import time
import multiprocessing

import pandas as pd
from tqdm import tqdm

from src.preprocessing.incidencias import Incidencias
from src.preprocessing.utils import find_best_match
from src.utils import load_config, get_logger, load_data


def find_best_match_wrapper(cod_articulo, articulos_codart):
    return find_best_match(cod_articulo, articulos_codart)


def find_best_matches_parallel(piezas: pd.DataFrame, articulos: pd.Series) -> pd.DataFrame:
    """
    Find the best match for each item in piezas in parallel.
    :param piezas: DataFrame with the data to "replace"
    :param articulos: Series of standard CODART to search
    :return: DataFrame with best matches and fuzzy scores
    """
    articulos_codart = articulos.to_list()  # List of 'CODART' to pass as an argument

    # Run the find_best_match function in parallel with a progress bar
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Pass both arguments to executor.map
        results = list(tqdm(
            executor.map(find_best_match_wrapper, piezas["cod_articulo"], [articulos_codart] * len(piezas)),
            total=len(piezas),
            desc="Processing Matches"
        ))

    # Convert results to DataFrame and assign columns
    return pd.DataFrame(results, columns=["CODART_A3", "Fuzzy_Score"])


def find_best_matches(
        env: str,
        input_articulos: str,
        input_piezas: str,
        output_best_matches: str
) -> None:
    config = load_config(file_name="config", env=env)
    data_config = load_config(file_name="data_config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start finding best matches in {env} environment")

    # Load data
    logger.info("Loading data")
    articulos = load_data(
        data_path=input_articulos,
        step_config=processing_config.processing.articulos,
    )

    piezas = load_data(
        data_path=input_piezas,
        step_config=processing_config.processing.piezas,
    )

    """
    Find the best match for the data in the elements_list
    :param elements_list: List of elements to search
    :return:
    """
    if piezas['cod_articulo'] is None:
        raise ValueError("Data is empty")

    logger.info("Finding best matches")
    """piezas[["CODART_A3", "Fuzzy_Score"]] = piezas["cod_articulo"].apply(
        lambda x: pd.Series(find_best_match(x, articulos['CODART']))
    )"""
    piezas.fillna("", inplace=True)
    piezas[['CODART_A3', 'Fuzzy_Score']] = find_best_matches_parallel(piezas, articulos['CODART'])

    logger.info("Saving best matches to disk")
    piezas[["cod_articulo", "CODART_A3", "Fuzzy_Score"]].to_csv(
        output_best_matches,
        quoting=csv.QUOTE_NONNUMERIC,
        index=False,
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
        "--input-articulos",
        type=str,
        help="Input articulos file"
    )
    parser.add_argument(
        "--input-piezas",
        type=str,
        help="Input piezas file"
    )
    parser.add_argument(
        "--output-best-matches",
        type=str,
        help="Output best matches file"
    )

    args = parser.parse_args()

    find_best_matches(
        env=args.env,
        input_articulos=args.input_articulos,
        input_piezas=args.input_piezas,
        output_best_matches=args.output_best_matches
    )
