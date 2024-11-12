import argparse
import time
import os
import traceback

from src.data_injection.a3 import load_a3_data
from src.db.connections import MySQLConnector, SqlServerConnector
from src.data_injection.myzone import load_myzone_data
from src.utils import load_config, get_logger


def fetch_data(env: str) -> None:
    if env not in ["dev", "prod"]:
        raise ValueError(
            f"Environment must be either dev or prod and got {env} instead"
        )

    config = load_config(file_name="config", env=env)
    data_config = load_config(file_name="data_config", env=env)
    logger = get_logger(config)

    mysql_conn = MySQLConnector(**config.database.myzone)
    myzone_data_to_fetch = data_config.data.myzone_data_to_fetch

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start fetching Myzone data in {env} environment")

    for data_to_fetch in myzone_data_to_fetch:
        try:
            logger.info(f"Start fetching {data_to_fetch} data")
            start_time = time.time()
            myzone_data = load_myzone_data(
                conn=mysql_conn,
                config=data_config,
                data=data_to_fetch
            )
            logger.info(f"{data_to_fetch} outputfile: {data_config.data[data_to_fetch].output_file}")
            myzone_data.to_csv(data_config.data[data_to_fetch].output_file, index=False)
            logger.info(f"Time elapsed for {data_to_fetch}: {time.time() - start_time}")
        except Exception as e:
            logger.error(f"Error loading {data_to_fetch}: {e}")
            logger.error(traceback.format_exc())

    a3_conn = SqlServerConnector(**config.database.a3)
    a3_data_to_fetch = data_config.data.a3_data_to_fetch

    for data_to_fetch in a3_data_to_fetch:
        try:
            logger.info(f"Start fetching {data_to_fetch} data")
            start_time = time.time()
            a3_data = load_a3_data(
                conn=a3_conn,
                config=data_config,
                data=data_to_fetch
            )
            logger.info(f"{data_to_fetch} outputfile: {data_config.data[data_to_fetch].output_file}")
            a3_data.to_csv(data_config.data[data_to_fetch].output_file, index=False)
            logger.info(f"Time elapsed for {data_to_fetch}: {time.time() - start_time}")
        except Exception as e:
            logger.error(f"Error loading {data_to_fetch}: {e}")
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch raw data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to fetch the data from",
        default="dev",
    )

    args = parser.parse_args()

    fetch_data(args.env)
