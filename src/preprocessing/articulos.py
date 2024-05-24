import os
from src.db.connections import SqlServerConnector


class Articulos:

    def __init__(self):

        self.a3_conn = SqlServerConnector(
            user="voliveira",
            password=os.environ.get("SQL_PASSWORD"),
            host="ROMPETECHOS",
            port="53373",
        )
        self.instancia = "REPLICA"
        self.data = None

    def get_articulos(self):
        self.__get_cod_artic()
        return self

    def get_unique_caracteristicas(self, numcar, caracteristicas):
        """
        Get the unique caracteristicas from the database
        :param numcar: Number of the caracteristica
        :param caracteristicas: Dataframe with the caracteristicas
        :return: self
        """
        return caracteristicas[
            (caracteristicas["NUMCAR"] == numcar) & (caracteristicas["TIPCAR"] == "A")
        ][["CODCAR", "DESCCAR"]]

    def __get_cod_artic(self):
        """
        Get the articles from the database
        :return: self
        """

        articulos = self.a3_conn.query_data(
            query="SELECT CODART, DESCART, CAR1, CAR2, CAR3, CAR4 FROM dbo.ARTICULO",
            database="Altra",
            instance=self.instancia,
        )

        caracteristicas = self.a3_conn.query_data(
            query="SELECT * FROM dbo.CARACTERISTICAS;",
            database="Altra",
            instance=self.instancia,
        )

        # Merge the caracteristicas
        for i in range(1, 5):
            articulos = articulos.merge(
                self.get_unique_caracteristicas(i, caracteristicas),
                left_on=f"CAR{i}",
                right_on="CODCAR",
                suffixes=(None, str(i)),
                how="left",
            )

        # Clean usuless columns
        self.data = articulos.drop(["CODCAR", "CODCAR2", "CODCAR3", "CODCAR4"], axis=1)
        # Rename to match patterns
        self.data = self.data.rename(columns={"DESCCAR": "DESCCAR1"})

        return self
