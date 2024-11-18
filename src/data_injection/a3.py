import pandas as pd
from omegaconf import DictConfig

from src.db.connections import SqlServerConnector


def load_a3_data(
    conn: SqlServerConnector, config: DictConfig, data: str
) -> pd.DataFrame:
    """
    Load data from SQLServer (a3) database
    :param config: DictConfig with the configuration
    :param conn: SQLServerConnector
    :param data: data (table) to load
    :return: DataFrame
    """

    query = ", ".join(config.data[data].columns)

    return conn.query_data(
        query=f"SELECT {query} " f"FROM dbo.{config.data[data].table_name}",
        database=config.data[data].database,
        instance=config.data[data].instance,
    )
