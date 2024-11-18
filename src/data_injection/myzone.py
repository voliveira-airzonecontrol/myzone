import pandas as pd
from omegaconf import DictConfig

from src.db.connections import MySQLConnector


def load_myzone_data(
    conn: MySQLConnector, config: DictConfig, data: str
) -> pd.DataFrame:
    """
    Load data from MySQL (myzone) database
    :param config: DictConfig with the configuration
    :param conn: MySQLConnector
    :param data: data to load
    :return: DataFrame
    """

    columns = ", ".join(config.data[data].columns)

    query = f"SELECT {columns} " f"FROM {config.data[data].table_name}"

    if "filter_date_column" in config.data[data]:
        query += (
            f" WHERE {config.data[data].filter_date_column} between "
            f"'{config.data[data].filter_date_start}' and '{config.data[data].filter_date_end}'"
        )

    return conn.query_data(query=query, database=config.data[data].database)
