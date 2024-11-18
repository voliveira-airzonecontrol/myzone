import pandas as pd
import os
import spacy
from bs4 import BeautifulSoup
import PyPDF2
import concurrent.futures
from tqdm import tqdm

from .incidencias import Incidencias


TRAINING_DATA_PATH = "..\\..\\Datos - Myzone\\TrainningData"
PRODUCT_DOCUMENTATION_PATH = (
    r"\\central4\Publica\Product_technical_documentation-Documentación_técnica_producto"
)
nlp = spacy.load("es_core_news_sm")
pd.options.mode.chained_assignment = None


class Corpus:
    def __init__(self, data_folder: str = "../DATA/"):
        self.incidencias = None
        self.faq = None
        self.products_docs = None
        self.catalogo = None
        self.data = None
        self.data_folder = data_folder

    def create_corpus(self) -> "Corpus":
        """
        Create the corpus
        :return: Self object
        """
        try:
            self.prepare_incidencias()
            self.prepare_faq()
            self.prepare_products_docs()
            self.prepare_catalogo()
        except Exception as e:
            print(f"Error creating corpus: {e}")
            raise e

        self.data = pd.concat(
            [self.incidencias, self.faq, self.products_docs, self.catalogo]
        )

        return self

    def load_corpus(self, file_path: str) -> "Corpus":
        """
        Load the corpus from a file
        :param file_path: File path
        :return: Self object
        """
        self.data = pd.read_csv(file_path)

        return self

    def save_corpus(self, file_path: str) -> None:
        """
        Save the corpus to a file
        :param file_path: File path
        :return: None
        """
        if self.data is None:
            raise ValueError("No data to save")

        self.data.to_csv(file_path, index=False, quoting=1)

    def prepare_incidencias(self, translation_date: str = "2024-05-09") -> "Corpus":
        """
        Prepare the incidencias data
        :param data_folder: Directory containing the translation files
        :param translation_date: Date of the translation files
        :return: Self object
        """

        incidencias = (
            Incidencias().get_incidencias(limit_date=translation_date).incidencias
        )

        self.incidencias = pd.DataFrame(
            {"text_to_analyse": incidencias["text_to_analyse"]}
        )

        return self

    def __remove_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def prepare_faq(self) -> "Corpus":
        """
        Prepare the FAQ data
        :param data_folder: Directory containing the FAQ data
        :return: Corpus object
        """
        faq = pd.read_csv(
            os.path.join(TRAINING_DATA_PATH, "FAQ.csv"), sep=";", header=None
        )
        faq.columns = ["text_to_analyse"]

        self.faq = faq["text_to_analyse"].apply(self.__remove_html_tags)

        return self

    def __get_pdf_files(self, path):
        pdf_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def __extract_text_from_pdf(self, pdf_path):
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

    def __process_pdf(self, pdf):
        # print(f'Processing {pdf}')
        try:
            df = self.__extract_text_from_pdf(pdf)
            return df
        except Exception as e:
            return pd.DataFrame()

    def prepare_products_docs(self) -> "Corpus":
        """
        Prepare the products documents data
        :return: Corpus object
        """
        pdfs = self.__get_pdf_files(PRODUCT_DOCUMENTATION_PATH)
        self.products_docs = pd.DataFrame()

        # Process the PDF files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map the process_pdf function to all the PDF files
            results = list(
                tqdm(executor.map(self.__process_pdf, pdfs), total=len(pdfs))
            )

            # Concatenate results as they complete
            for result in results:
                self.products_docs = pd.concat([self.products_docs, result])

        return self

    def prepare_catalogo(self) -> "Corpus":
        """
        Prepare the catalogo data
        :return: Corpus object
        """
        self.catalogo = self.__extract_text_from_pdf(
            os.path.join(TRAINING_DATA_PATH, "catalogo.pdf")
        )

        return self
