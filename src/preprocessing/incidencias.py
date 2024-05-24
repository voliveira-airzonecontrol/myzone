import os
from src.db.connections import MySQLConnector
import pandas as pd
from typing import Optional


class Incidencias:
    def __init__(
            self,
            data_folder: str = "../DATA/"
    ):

        self.myzone_conn = MySQLConnector(
            user="readmyzone",
            password=os.environ.get("MYSQL_PASSWORD"),
            host="192.168.2.7",
            port="3306",
        )

        self.data = None
        self.data_folder = data_folder

    def get_incidencias(
            self,
            limit_date: str = "2024-05-09"
    ) -> 'Incidencias':

        self.__get_incidencia_data(limit_date=limit_date)

        return self

    def __load_incidencias_data(
            self,
    ) -> tuple[Optional[pd.DataFrame], ...]:
        """
        Get the data from the tables sav_incidencias, sav_piezas, sav_estados, and sav_incidencias_tipo
        :return: Tuple with the data from the tables
        """
        tables = ["sav_incidencias", "sav_piezas", "sav_estados", "sav_incidencias_tipo"]
        return tuple(self.myzone_conn.query_data(f"SELECT * FROM {table}", database="myzone") for table in tables)

    def __load_text_to_translate(self, data_folder: str) -> dict:
        """
        **UNUSED**
        Load translation data from CSV files
        :param data_folder: Directory containing the translation CSV files
        :return: Dictionary with translation dataframes
        """
        fields_to_translate = ["desc_problema", "problema", "descripcion"]
        text_to_translate = {}
        for text in fields_to_translate:
            text_to_translate[text] = pd.read_csv(
                os.path.join(data_folder, f"{text}.csv"),
                sep="¬",
                encoding="utf-8-sig"
            )
        return text_to_translate

    def __load_translation_data(self, data_folder: str) -> dict:
        """
        Clean and load translated text data from CSV files
        :param data_folder: Directory containing the translated CSV files
        :return: Dictionary with cleaned translated dataframes
        """
        translations = ["desc_problema_translated", "descripcion_translated", "problema_translated"]
        cleaned_data = {}

        for trans in translations:
            df = pd.read_csv(
                os.path.join(data_folder, f"{trans}.csv"),
                sep="¬",
                encoding="utf-8-sig",
                engine="python"
            )
            df = df[~df[trans].isin([trans])]
            cleaned_data[trans] = df

        return cleaned_data

    def __get_incidencia_data(
            self,
            limit_date: str = "2024-05-09"
    ) -> 'Incidencias':
        """
        Get the incidencias data
        :return: Data from the incidencias table
        """
        # Get the data
        sav_incidencias, sav_piezas, sav_estados, sav_incidencias_tipo = self.__load_incidencias_data()

        # Merge the data
        dataset = sav_incidencias.merge(
            sav_piezas,
            left_on="codigo",
            right_on="codigo_incidencia",
            how="left",
            suffixes=(None, "_pieza"),
        )

        dataset = dataset.merge(
            sav_estados,
            left_on="estado",
            right_on="id",
            how="left",
            suffixes=(None, "_estado"),
        )

        dataset = dataset.merge(
            sav_incidencias_tipo,
            left_on="tipo",
            right_on="id",
            how="left",
            suffixes=(None, "_tipo"),
        )

        # Convert the modification_date to datetime
        dataset["modification_date"] = pd.to_datetime(
            dataset["modification_date"], errors="coerce"
        )

        # Filter the data and make sure:
        # 1. The modification_date is before the date of the translation files
        # 2. The tipo is 1 (Garantia)
        # 3. The estado is 2 (Validado) or 6 (Cerrado)
        clean_dataset = dataset[
            (dataset["tipo"] == 1)
            & (dataset["estado"].isin([2, 6]))
            & (dataset["modification_date"] < limit_date)
            ]

        # Load from disk the text to translate dictionary
        translation_data = self.__load_translation_data(self.data_folder)

        # Merge the translated text with the original dataset
        for field in ["desc_problema", "descripcion", "problema"]:
            clean_dataset = clean_dataset.merge(
                translation_data[field + "_translated"], left_on=field, right_on=field, how="left"
            )
            clean_dataset.fillna({field + "_translated": clean_dataset[field]}, inplace=True)

        # Create final dataset
        clean_dataset.fillna("", inplace=True)

        # The text_to_analyse field will be defined by the usecase (dependent on the model)
        """clean_dataset["text_to_analyse"] = clean_dataset[[
            "desc_problema_translated",
            "descripcion_translated",
            "problema_translated",
            "cod_articulo"
        ]].apply(lambda x: " ".join(x), axis=1)"""

        self.data = clean_dataset

        return self
