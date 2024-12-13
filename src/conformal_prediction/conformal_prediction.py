import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.training.utils import train_model
from src.utils import load_config, get_logger, load_data, save_data

from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score


def conformal_prediction(
    env: str,
    input_dataset: str,
    input_classification_model: str,
    output_conformal_prediction_model: str,
    model_type: str = "XGBoost",
) -> None:
    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start training the classifier in {env} environment")

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

    # Load classification model
    logger.info(f"Load classification model")
    clf = joblib.load(f"../../MODELS/model.joblib")

    if model_type == "XGBoost":
        X_test = np.vstack(X_test[training_config.training[model_type].features].values)
        y_test = y_test.values
        X_cp = np.vstack(X_cp[training_config.training[model_type].features].values)
        y_cp = y_cp.values

    # Train conformal prediction model
    logger.info(f"Train conformal prediction model")
    mapie_score = MapieClassifier(estimator=clf, cv="prefit", method="score")
    mapie_score.fit(X_cp, y_cp)

    alpha = [0.2, 0.1, 0.05]
    y_pred_score, y_ps_score = mapie_score.predict(X_test, alpha=alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run conformal prediction")
    parser.add_argument(
        "--env",
        type=str,
        help="Environment to preprocess the data from",
        default="dev",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        help="Input dataset",
    )
    parser.add_argument(
        "--input-classification-model",
        type=str,
        help="Input model",
    )
    parser.add_argument(
        "--output-conformal-prediction-model",
        type=str,
        help="Output conformal prediction model",
    )
    args = parser.parse_args()

    conformal_prediction(
        env=args.env,
        input_dataset=args.input_dataset,
        input_classification_model=args.input_classification_model,
        output_conformal_prediction_model=args.output_conformal_prediction_model,
    )
