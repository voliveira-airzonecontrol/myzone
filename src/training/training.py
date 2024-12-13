import argparse

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.training.utils import train_model
from src.utils import load_config, get_logger, load_data, save_data


def classifier_training(
    env: str, input_dataset: str, output_model: str, model_type: str = "XGBoost"
) -> None:
    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start training the classifier in {env} environment")

    # Load dataset
    """X = load_data(
        data_path=input_dataset,
        step_config=training_config.training.training_data,
    )"""
    X = pd.read_parquet(input_dataset)

    # Split dataset into training, test and conformal prediction
    logger.info(f"Split dataset into training, test and conformal prediction")
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop(columns=[training_config.training[model_type].target]),
        X[training_config.training[model_type].target],
        test_size=training_config.training[model_type].test_size,
        random_state=training_config.training.random_state,
    )
    X_test, X_cp, y_test, y_cp = train_test_split(
        X_test,
        y_test,
        test_size=training_config.training[model_type].cp_size,
        random_state=training_config.training.random_state,
    )

    # Train classifier
    logger.info(f"Train classifier")
    best_model = train_model(
        X_train[training_config.training[model_type].features],
        y_train,
        X_test[training_config.training[model_type].features],
        y_test,
        training_config,
        model_type=model_type,
    )

    # Dump best model with joblib
    logger.info(f"Save best model")
    with open(output_model, "wb") as f:
        joblib.dump(best_model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to preprocess the data from",
        default="dev",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        help="Path to the input dataset",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        help="Path to the output model",
    )

    args = parser.parse_args()

    classifier_training(
        env=args.env,
        input_dataset=args.input_dataset,
        output_model=args.output_model,
    )
