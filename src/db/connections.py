from abc import abstractmethod, ABC
from typing import Union

import pyodbc
from sqlalchemy import create_engine
import pandas as pd


# Abstract class for the connector
class Connector(ABC):
    """
    Abstract class for the connector to the Airzone databases
    """
    def __init__(self, user, password, host, port):
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    @abstractmethod
    def query_data(self, query: str, database: str) -> Union[pd.DataFrame, None]:
        pass


class SqlServerConnector(Connector):
    """
    Connector class for the SQL Server databases
    """
    def __init__(self, user, password, host, port):
        super().__init__(user, password, host, port)

    def query_data(self, query: str, database: str, instance: str = None) -> Union[pd.DataFrame, None]:
        """
        Query the data from the SQL Server database
        :param query: Query to execute
        :param database: Database name
        :param instance: Instance name (if any)
        :return: Data from the query (if error None)
        """
        if not database:
            print('Database name is required')
            return None

        if instance:
            conn_str = f"DRIVER={{SQL Server}};SERVER={self.host}\\{instance},{self.port};DATABASE={database};UID={self.user};PWD={self.password}"
        else:
            conn_str = f"DRIVER={{SQL Server}};SERVER={self.host},{self.port};DATABASE={database};UID={self.user};PWD={self.password}"

        # Create the connection
        try:
            conn = pyodbc.connect(conn_str)
        except Exception as e:
            print(f'Error creating connection: {e}')
            return None

        # query the data
        try:
            data = pd.read_sql(query, conn)
        except Exception as e:
            print(f'Error: {e}')
            data = None

        return data


class MySQLConnector(Connector):
    """
    Connector class for the MySQL databases
    """
    def __init__(self, user, password, host, port):
        super().__init__(user, password, host, port)

    def query_data(self, query: str, database: str) -> Union[pd.DataFrame, None]:
        """
        Query the data from the MySQL database
        :param query: Query to execute
        :param database: Database name
        :return: Data from the query (if error None)
        """

        # Create the connection string
        conn_str = f'mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{database}'

        # Create the engine
        engine = create_engine(conn_str)

        # Query the data
        try:
            data = pd.read_sql(query, engine)
        except Exception as e:
            print(f'Error: {e}')
            return None

        return data


class OracleConnector(Connector):
    """
    Connector class for the Oracle databases
    """
    def __init__(self, user, password, host, port):
        super().__init__(user, password, host, port)

    def query_data(self, query: str, database: str) -> Union[pd.DataFrame, None]:
        """
        Query the data from the Oracle database
        :param query: Query to execute
        :param database: Database name
        :return: Data from the query (if error None)
        """
        # Define your connection string
        conn_str = f'oracle+cx_oracle://{self.user}:{self.password}@{self.host}:{self.port}/{database}'

        # Create the engine
        engine = create_engine(conn_str)

        # Query the data
        try:
            data = pd.read_sql(query, engine)
        except Exception as e:
            print(f'Error: {e}')
            return None

        return data
