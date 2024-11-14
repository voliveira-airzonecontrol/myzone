import argparse
import pandas as pd
# import spacy
# import gensim.models.doc2vec
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from src.preprocessing.dataset import Dataset
from src.preprocessing.incidencias import Incidencias
from src.preprocessing.utils import pre_process_text_spacy

# assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

from src.utils import load_config, get_logger, load_data, save_data

# nlp = spacy.load("es_core_news_sm")

'''
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
'''


def preprocess(
        env: str,
        translation_data_folder: str,
        raw_data_folder: str,
        input_articulos: str,
        input_best_matches: str,
        output_path: str,
) -> None:
    config = load_config(file_name="config", env=env)
    data_config = load_config(file_name="data_config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start preprocessing data in {env} environment")

    # Load data
    logger.info(f"Load data")
    articulos = load_data(
        data_path=input_articulos,
        step_config=processing_config.processing.articulos,
    )

    logger.info("Processing incidencias")
    incidencias = (
        Incidencias(
            translation_data_folder=translation_data_folder,
            raw_data_folder=raw_data_folder,
            config=processing_config,
        )
        .get_incidencias()
        .load_best_match(input_best_matches)
        .data
    )

    logger.info("Generate dataset")
    clean_dataset = Dataset(incidencias, articulos).generate_dataset().data

    logger.info("Preprocess text")
    clean_dataset["processed_text_to_analyse"] = pre_process_text_spacy(
        clean_dataset["text_to_analyse"].values,
        **processing_config.processing['tokenizer']
    )

    logger.info(f"Save preprocessed data with shape {clean_dataset.shape}")
    save_data(clean_dataset, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to preprocess the data from",
        default="dev",
    )
    parser.add_argument(
        "--translation-data-folder",
        type=str,
        help="Folder with the translation data",
    )
    parser.add_argument(
        "--raw-data-folder",
        type=str,
        help="Folder with the raw data",
    )
    parser.add_argument(
        "--input-articulos",
        type=str,
        help="Path to the articulos data",
    )
    parser.add_argument(
        "--input-best-matches",
        type=str,
        help="Path to the best matches data",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output data",
    )

    args = parser.parse_args()

    preprocess(
        env=args.env,
        translation_data_folder=args.translation_data_folder,
        raw_data_folder=args.raw_data_folder,
        input_articulos=args.input_articulos,
        input_best_matches=args.input_best_matches,
        output_path=args.output_path,
    )
