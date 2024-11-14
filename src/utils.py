from __future__ import annotations

import csv
import logging.config
import os
from typing import Dict, Optional

import numpy as np
import yaml

import pandas as pd
from omegaconf import OmegaConf, DictConfig, ListConfig


def replace_nans(data: pd.Series) -> pd.Series:
    """Replace NaNs and other specific string representations with None."""
    return data.replace({"nan": None, "NaT": None, "None": None, "<NA>": None})


def convert_dtypes(data: pd.DataFrame, step_config: DictConfig) -> pd.DataFrame:
    """
    Convert data types of columns based on provided step configuration.
    """

    def convert_to_float(value: str) -> float:
        if isinstance(value, str):  # Check if the value is a string
            value = value.replace(",", ".")  # Replace comma with dot
        try:
            return float(value)  # Convert to float
        except (ValueError, TypeError):
            return None  # Return None for non-numeric values

    type_mapping = {
        "int": lambda col: pd.to_numeric(data[col], errors="coerce").astype("Int64"),
        "float": lambda col: pd.to_numeric(data[col], errors="coerce"),
        "float_especial": lambda col: data[col].apply(convert_to_float),
        "datetime64": lambda col: pd.to_datetime(data[col], errors="coerce"),
        "str": lambda col: (
            replace_nans(data[col].astype(str, errors="ignore"))
            if data[col].dtype != "float"
            else replace_nans(
                pd.to_numeric(data[col], errors="coerce").astype("Int64").astype(str)
            )
        ),
    }

    for col, dtype in step_config.dtypes.items():
        try:
            if dtype in type_mapping:
                data[col] = type_mapping[dtype](col)
            else:
                # Attempt to cast using the dtype if not handled above
                data[col] = data[col].astype(dtype, errors="ignore")
        except Exception as e:
            print(f"Error converting column {col}: {e}")

    return data


def load_data(data_path: str, step_config: DictConfig) -> pd.DataFrame:
    """
    Load and convert data from a CSV file
    """
    return convert_dtypes(
        pd.read_csv(data_path, low_memory=False, na_values=["", "<NA>"]),
        step_config=step_config,
    )


def save_data(data: pd.DataFrame, output_path: str) -> None:
    """
    Save data to a CSV file
    """
    # Extract the directory from the output path
    output_dir = os.path.dirname(output_path)

    # Create directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)


def get_logger(config: DictConfig, path: Optional[str] = None) -> logging.Logger:
    """
    Get the logger object
    """

    config_path = path if path else os.path.join("config", config.environment, "config.yaml")

    # Load and configure the logger using YAML logging configuration
    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)  # Load YAML config file for logging
        logging.config.dictConfig(yaml_config["logging"])  # Apply logging configuration

    return logging.getLogger("pipeline_logger")


def load_config(file_name: str, env: str = "dev", folder: str = None) -> DictConfig | ListConfig:
    """
    Load the configuration file

    :param file_name: File name of the configuration file
    :param env: Environment to load the configuration file
    :param folder: Folder where the configuration files are stored
    :return: Configuration object
    """
    if folder is None:
        folder = "config"

    file_path = os.path.join(folder, env, f"{file_name}.yaml")

    return OmegaConf.load(file_path)
