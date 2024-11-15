import argparse
import concurrent.futures
import csv
import os
import time
import multiprocessing
import concurrent.futures

from psutil._compat import unicode
from tqdm import tqdm
import pandas as pd
import spacy
from langdetect import detect
from langdetect import DetectorFactory
from bs4 import BeautifulSoup
import PyPDF2

DetectorFactory.seed = 0
nlp = spacy.load("es_core_news_sm")

from src.utils import load_config, get_logger, load_data, save_data


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def get_pdf_files(path):
    pdf_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file).encode('utf-8').decode('utf-8'))
    return pdf_files


def extract_text_from_pdf(pdf_path):
    text = []
    sentences = []
    try:
        with open(pdf_path, "rb") as file:
            pdf = PyPDF2.PdfReader(file)
            for page in range(len(pdf.pages)):
                text.append(pdf.pages[page].extract_text())

        for i, page in enumerate(text):
            doc = nlp(page)
            for sentence in doc.sents:
                sentences.append(sentence.text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        raise e

    return pd.DataFrame(sentences, columns=["text_to_analyse"])


def process_pdf(pdf):
    # print(f'Processing {pdf}')
    try:
        df = extract_text_from_pdf(pdf)
        return df
    except Exception as e:
        return pd.DataFrame()


def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def generate_corpus(
        env: str,
        documentation_path: str,
        training_data_path: str,
        preprocessed_data_path: str,
        output_corpus: str
) -> None:

    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start generating corpus in {env} environment")

    # Load preprocessed data
    logger.info(f"Load preprocessed data")
    preprocessed_data = load_data(
        data_path=preprocessed_data_path,
        step_config=processing_config.processing.preprocessed_data,
    )

    incidencias = preprocessed_data[["text_to_analyse"]]

    logger.info("Load FAQ data")
    faq_path = os.path.join(training_data_path, "FAQ.csv")
    faq = pd.read_csv(faq_path, sep=";", header=None)
    faq.columns = ["text_to_analyse"]

    # Remove html tags
    faq["text_to_analyse"] = faq["text_to_analyse"].apply(remove_html_tags)

    logger.info("Load product data")
    pdfs = get_pdf_files(documentation_path)
    product_documentation = pd.DataFrame()
    # Process the PDF files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the process_pdf function to all the PDF files
        results = list(tqdm(executor.map(process_pdf, pdfs), total=len(pdfs)))

        # Concatenate results as they complete
        for result in results:
            product_documentation = pd.concat([product_documentation, result])
    

    logger.info("Load product catalog data")
    catalogo_path = os.path.join(training_data_path, "catalogo.pdf")
    catalogo = extract_text_from_pdf(catalogo_path)

    corpus = pd.concat([incidencias, faq, catalogo, product_documentation])

    logger.info("Detecting language")
    corpus["language"] = corpus["text_to_analyse"].apply(detect_language)
    corpus = corpus[corpus["language"].isin(["es", "pt"])]

    logger.info(f"Save corpus with shape {corpus.shape}")
    save_data(corpus, output_corpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to preprocess the data from",
        default="dev",
    )
    parser.add_argument(
        "--input-documentation-path",
        type=str,
        help="Input articulos file"
    )
    parser.add_argument(
        "--input-training-data-path",
        type=str,
        help="Input piezas file"
    )

    parser.add_argument(
        "--input-preprocessed-data",
        type=str,
        help="Input preprocessed data file"
    )
    parser.add_argument(
        "--output-corpus",
        type=str,
        help="Output best matches file"
    )

    args = parser.parse_args()

    generate_corpus(
        env=args.env,
        documentation_path=args.input_documentation_path,
        training_data_path=args.input_training_data_path,
        preprocessed_data_path=args.input_preprocessed_data,
        output_corpus=args.output_corpus,
    )
