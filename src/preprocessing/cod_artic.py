import os
from src.db.connections import SqlConnector

class CodArtic():

    def __init__(self):

        self.connector = SqlConnector(
            user='voliveira',
            password=os.environ.get('SQL_PASSWORD'),
            host='ROMPETECHOS',
            port='53373'
        )

    def get_cod_artic(self):
        pass

    def __get_cod_artic(self):
        pass