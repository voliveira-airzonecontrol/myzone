import logging
from abc import abstractmethod, ABC
from typing import Union

import cx_Oracle
import pyodbc
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
import jaydebeapi
import clickhouse_connect

# Configure logging
logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

    def query_data(
        self, query: str, database: str, instance: str = None
    ) -> Union[pd.DataFrame, None]:
        """
        Query the data from the SQL Server database
        :param query: Query to execute
        :param database: Database name
        :param instance: Instance name (if any)
        :return: Data from the query (if error None)
        """
        if not database:
            print("Database name is required")
            return None

        if instance:
            conn_str = f"DRIVER={{SQL Server}};SERVER={self.host}\\{instance},{self.port};DATABASE={database};UID={self.user};PWD={self.password}"
        else:
            conn_str = f"DRIVER={{SQL Server}};SERVER={self.host},{self.port};DATABASE={database};UID={self.user};PWD={self.password}"

        # Create the connection
        try:
            conn = pyodbc.connect(conn_str)
        except Exception as e:
            print(f"Error creating connection: {e}")
            return None

        # query the data
        try:
            data = pd.read_sql(query, conn)
        except Exception as e:
            print(f"Error: {e}")
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
        conn_str = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{database}"

        # Create the engine
        engine = create_engine(conn_str)

        # Query the data
        try:
            data = pd.read_sql(query, engine)
        except Exception as e:
            print(f"Error: {e}")
            return None

        return data


class OracleConnector:
    """
    Connector class for the Oracle databases
    """

    def __init__(self, user, password, host, port, environment: str = "dev", **kwargs):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.environment = environment
        self.driver_path = kwargs["driver_path"] if "driver_path" in kwargs else None

    def query_data(
        self, query: str, database: str, parse_dates: Union[list, dict] = None
    ) -> Union[pd.DataFrame, None]:
        """
        Query the data from the Oracle database
        :param query: Query to execute
        :param database: Database name
        :param parse_dates: Columns to parse as dates
        :return: Data from the query (if error None)
        """
        if self.environment == "dev":
            # Define your connection string
            conn_str = f"oracle+cx_oracle://{self.user}:{self.password}@{self.host}:{self.port}/{database}"
            # Create the engine
            engine = create_engine(conn_str)

            # Query the data
            try:
                # Query the data
                data = pd.read_sql(query, engine, parse_dates=parse_dates)
                return data
            except cx_Oracle.DatabaseError as db_err:
                (error,) = db_err.args
                logging.error(f"Database error: {error.code} - {error.message}")
            except SQLAlchemyError as sa_err:
                logging.error(f"SQLAlchemy error: {sa_err}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")

        elif self.environment == "prod":
            if not self.driver_path:
                logging.error("Driver path is required")
                return None

            # Define your connection string
            conn_str = f"jdbc:oracle:thin:@{self.host}:{self.port}/{database}"

            # Create the connection
            conn = jaydebeapi.connect(
                jclassname="oracle.jdbc.driver.OracleDriver",
                url=conn_str,
                driver_args={"user": self.user, "password": self.password},
                jars=self.driver_path,
            )
            # Query the data
            try:
                # Query the data
                cursor = conn.cursor()
                # cursor.execute("ALTER SESSION SET TIME_ZONE = 'UTC'")  # Change to your preferred time zone
                cursor.execute(query)
                data = cursor.fetchall()
                # Get column names from cursor.description
                columns = [col[0] for col in cursor.description]
                cursor.close()
                conn.close()
                return pd.DataFrame(data, columns=columns)
            except Exception as e:
                logging.error(f"Error: {e}")

        return None


class ClickHouseConnector(Connector):
    """
    Connector class for the ClickHouse databases
    """

    def __init__(self, user, password, host, port=None):
        super().__init__(user, password, host, port)

    def query_data(self, query: str, database: str = "") -> Union[pd.DataFrame, None]:
        """
        Query the data from the ClickHouse database
        :param query: Query to execute with the database name included
        :return: Data from the query (if error None)
        """

        # Query the data
        try:
            client = clickhouse_connect.get_client(
                host=self.host,
                user=self.user,
                password=self.password,
                secure=True
            )

            data = client.query(query)

            return pd.DataFrame(data.result_rows, columns=data.column_names)

        except Exception as e:
            logging.error(f"Error: {e}")
            return None
