import os
from src.db.connections import SqlServerConnector


class Caracteristicas:

    def __init__(self):

        self.a3_conn = SqlServerConnector(
            user="voliveira",
            password=os.environ.get("SQL_PASSWORD"),
            host="ROMPETECHOS",
            port="53373",
        )
        self.instancia = "REPLICA"
        self.data = None

    def get_caracteristicas(self):
        self.__get_cod_caracteristicas()
        return self

    def __get_cod_caracteristicas(self):
        """
        Get the articles from the database
        :return: self
        """

        caracteristicas = self.a3_conn.query_data(
            query="SELECT * FROM dbo.CARACTERISTICAS;",
            database="Altra",
            instance=self.instancia,
        )

        self.data = caracteristicas
        return self