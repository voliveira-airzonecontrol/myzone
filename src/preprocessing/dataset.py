import pandas as pd

# Define the families to remove
SPECIAL_FAMILIES = [80, 98]
MEANINGLESS_FAMILIES = [
    79, 78, 96, 71, 25, 73, 75, 61,
    18, 27, 26, 79, 34, 78, 30, 54,
    72, 32, 44, 29, 82, 81, 62, 51,
    65, 59, 21, 60, 77, 43, 48, 20,
    41, 28, None, "",
]


class Dataset:

    def __init__(self, incidencias: pd.DataFrame, articulos: pd.DataFrame):
        self.data = None
        self.incidencias = incidencias
        self.articulos = articulos

    def generate_dataset(self, threshold: float = 85) -> "Dataset":
        """
        Generate the dataset
        :param threshold: Threshold to filter the fuzzy score
        :return: self
        """

        self.data = self.incidencias.merge(
            self.articulos, left_on="CODART_A3", right_on="CODART", how="left"
        )

        # Generate the text to analyse
        self.data["text_to_analyse"] = self.data[
            [
                "desc_problema_translated",
                "descripcion_translated",
                "problema_translated",
            ]
        ].apply(lambda x: " ".join(x), axis=1)

        # Fill na values
        self.data.loc[:, "CAR3"] = self.data["CAR3"].fillna("")
        self.data.loc[:, "CAR2"] = self.data["CAR2"].fillna("")

        # Remove the meanless families
        self.data = self.data[~self.data["CAR3"].isin(MEANINGLESS_FAMILIES)]
        # Remove the special families
        self.data = self.data[~self.data["CAR3"].isin(SPECIAL_FAMILIES)]

        # Remove tiny descriptions
        """
            descriptions with less than 25 characters are removed
            descriptions with less than 25 characters after removing the "NO FUNCIONA" string are removed
        """
        self.data = self.data[self.data["text_to_analyse"].str.len() > 25]
        self.data = self.data[
            self.data["text_to_analyse"].str.replace("NO FUNCIONA", "").str.len() > 25
        ]

        # Clean low similariy scores
        self.data = self.data[self.data["Fuzzy_Score"] >= threshold]

        return self
