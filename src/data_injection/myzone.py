import pandas as pd
from omegaconf import DictConfig

from src.db.connections import MySQLConnector


def load_myzone_data(
        conn: MySQLConnector,
        config: DictConfig,
        data: str
) -> pd.DataFrame:
    """
    Load data from MySQL (myzone) database
    :param config: DictConfig with the configuration
    :param conn: MySQLConnector
    :param data: data to load
    :return: DataFrame
    """

    query = ', '.join(config.data[data].columns)

    return conn.query_data(
        query= f"SELECT {query} "
               f"FROM {config.data[data].table_name}",
        database=config.data[data].database
    )
