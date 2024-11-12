import argparse
import concurrent.futures
import csv
import os
import time

import pandas as pd
from deep_translator import GoogleTranslator
from langdetect import detect
from langdetect import DetectorFactory
from tqdm import tqdm

DetectorFactory.seed = 0

from src.utils import load_config, get_logger, load_data


# Detect language of the texts
def detect_language(text):
    if len(str(text)) < 5:
        return "Too short"

    try:
        return "es" if str(text) == "" else detect(str(text).lower())
    except Exception as e:
        return "Error"


# Function to apply detect_language in parallel
def detect_language_parallel(df, field):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Apply detect_language in parallel
        results = list(executor.map(detect_language, df[field]))
    return results


# Main function to detect language in parallel for each field
def detect_language_for_fields(text_to_translate, fields_to_translate, logger):
    for field in fields_to_translate:
        logger.info(f"Detecting language for field: {field}")
        # Apply parallel language detection and store results in a new column
        text_to_translate[field][f"{field}_lg"] = detect_language_parallel(
            text_to_translate[field], field
        )


def translate_in_batches(
        df: pd.DataFrame,
        column: str,
        folder_path="./output_data/dev",
        batch_size=10):
    """
    Function to translate the text in batches.
    It uses the Google Translator API to translate the text.
    It writes the translated text to a csv file.
    :param folder_path: path to the folder where the translated text will be saved
    :param df: dataframe with the text to translate
    :param column: column to translate
    :param batch_size: size of the batch to translate
    :return: None
    """
    total_rows = len(df)
    with tqdm(total=total_rows) as pbar:
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i: i + batch_size][column].tolist()
            try:
                translations = GoogleTranslator(
                    source="auto", target="es"
                ).translate_batch(batch)
                # Test if csv file exists, if not created it and add the translated text to it
                if not os.path.exists(os.path.join(folder_path, f"{column}_translated.csv")):
                    pd.DataFrame(
                        {column: batch, f"{column}_translated": translations}
                    ).to_csv(
                        os.path.join(folder_path, f"{column}_translated.csv"),
                        mode="a",
                        index=False,
                        quoting=csv.QUOTE_NONNUMERIC
                    )
                else:
                    pd.DataFrame(
                        {column: batch, f"{column}_translated": translations}
                    ).to_csv(
                        os.path.join(folder_path, f"{column}_translated.csv"),
                        mode="a",
                        header=False,
                        index=False,
                        quoting=csv.QUOTE_NONNUMERIC
                    )
                pbar.update(len(batch))
            except Exception as e:
                print(f"Error occurred during translation: {e}")


def translate(
        env: str,
        input_sav_incidencias: str,
        input_sav_piezas: str,
        input_sav_estados: str,
        input_sav_incidencias_tipo: str,
        output_path: str
) -> None:
    config = load_config(file_name="config", env=env)
    data_config = load_config(file_name="data_config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start translating data in {env} environment")

    # Load data
    logger.info("Loading data")
    incidencias = load_data(
        data_path=input_sav_incidencias,
        step_config=processing_config.processing.incidencias,
    )
    piezas = load_data(
        data_path=input_sav_piezas,
        step_config=processing_config.processing.piezas,
    )
    estados = load_data(
        data_path=input_sav_estados,
        step_config=processing_config.processing.estados,
    )
    incidencias_tipo = load_data(
        data_path=input_sav_incidencias_tipo,
        step_config=processing_config.processing.incidencias_tipo,
    )

    logger.info("Merging myzone data")
    dataset = incidencias.merge(
        piezas,
        left_on="codigo",
        right_on="codigo_incidencia",
        how="left",
        suffixes=(None, "_pieza"),
    )
    dataset = dataset.merge(
        estados, left_on="estado", right_on="id", how="left", suffixes=(None, "_estado")
    )
    dataset = dataset.merge(
        incidencias_tipo,
        left_on="tipo",
        right_on="id",
        how="left",
        suffixes=(None, "_tipo"),
    )

    clean_dataset = dataset[(dataset["tipo"] == 1) & (dataset["estado"].isin([2, 6]))]

    fields_to_translate = ["desc_problema", "problema", "descripcion"]
    text_to_translate = {}

    # Get unique values for each field
    for field in fields_to_translate:
        text_to_translate[field] = pd.DataFrame(
            clean_dataset[field].unique(), columns=[field]
        )

    # Detect language
    """logger.info("Detecting language")
    for field in fields_to_translate:
        text_to_translate[field][f"{field}_lg"] = text_to_translate[field][field].apply(
            detect_language
        )"""

    # Run the language detection in parallel
    detect_language_for_fields(text_to_translate, fields_to_translate, logger)

    # Translate texts that are not spanish text
    logger.info("Translating texts")
    for text in text_to_translate.keys():
        translate_in_batches(
            df=text_to_translate[text][
                ~text_to_translate[text][f"{text}_lg"].isin(["es", "Error", "Too short"])
            ],
            column=text,
            folder_path=output_path,
            batch_size=50,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch raw data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment",
        default="dev",
    )
    parser.add_argument(
        "--input-incidencias",
        type=str,
        help="Input sav incidencias file",
    )
    parser.add_argument(
        "--input-piezas",
        type=str,
        help="Input sav piezas file",
    )
    parser.add_argument(
        "--input-estados",
        type=str,
        help="Input sav estados file",
    )
    parser.add_argument(
        "--input-incidencias-tipo",
        type=str,
        help="Input sav incidencias tipo file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path",
    )

    args = parser.parse_args()

    translate(
        env=args.env,
        input_sav_incidencias=args.input_incidencias,
        input_sav_piezas=args.input_piezas,
        input_sav_estados=args.input_estados,
        input_sav_incidencias_tipo=args.input_incidencias_tipo,
        output_path=args.output_path
    )
