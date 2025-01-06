from itertools import product
from typing import Any, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.utils import resample
from xgboost import XGBClassifier
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm


def log_to_mlflow(
    run_name: str,
    params: dict,
    metrics: dict,
    model_name: str,
    model: Any,
    model_type: str,
    artifacts: Optional[dict] = None,
    experiment_name: str = "Default Experiment",
    tracking_uri: str = "http://192.168.2.241:5000",
    nested: bool = False,
) -> None:
    """
    Log the model to MLflow.
    :param model_type: Model type (sklearn, keras, pytorch, xgboost)
    :param run_name: Run name
    :param model_name: Model name
    :param params: Parameters
    :param metrics: Metrics
    :param model: Model
    :param experiment_name:
    :param tracking_uri:
    :param nested:
    :return: None
    """

    # Set the tracking URI to the specified MLflow server
    mlflow.set_tracking_uri(tracking_uri)

    # Create or set the experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if model:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path=model_type)
            elif model_type == "keras":
                mlflow.keras.log_model(model, artifact_path=model_type)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, artifact_path=model_type)
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, artifact_path=model_type)
            else:
                raise ValueError(f"Model type {model_type} not supported")

            # Register the model in the Model Registry with a specific name
            model_uri = f"runs:/{run.info.run_id}/{model_type}"
            model_version = mlflow.register_model(model_uri, name=model_name)

            print(
                f"Model {model_version.name} registered with version: {model_version.version}"
            )

        if artifacts:
            for key, value in artifacts.items():
                mlflow.log_artifact(value, key)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: DictConfig,
    model_type: str,
) -> Any:
    """
    Function to grid search hyperparameters for a given model type.
    Logs each run to MLflow.
    :param X: Input features
    :param y: Target variable
    :param X_test: Test features
    :param y_test: Test target
    :param config: Omegaconf Configuration
    :param model_type: Model type (XGBoost, RandomForest, SVC)
    :return: Best model
    """

    models = {
        "XGBoost": XGBClassifier,
        "RandomForest": RandomForestClassifier,
        "SVC": SVC,
    }

    if model_type not in models:
        raise ValueError(f"Model type {model_type} not supported")

    model_config = config.training[model_type]

    model_params = model_config.params_list

    params = {}
    for param in model_params:
        params[param] = model_config[param]

    param_combinations = list(product(*params.values()))

    # Create DMatrix for XGBoost
    if model_type == "XGBoost":
        X = np.vstack(X.values)
        y = y.values
        X_test = np.vstack(X_test.values)
        y_test = y_test.values
    else:
        X = pd.DataFrame(X.tolist())
        y = y.values
        X_test = pd.DataFrame(X_test.tolist())
        y_test = y_test.values

    # Initialize the best metrics and model
    best_metrics = {
        "accuracy": 0,
        "f1": 0,
        "precision": 0,
        "recall": 0,
        "roc_auc": 0,
        "cohen_kappa": 0,
        "mean_test_score": 0,
        "std_test_score": 0,
        "fit_time": 0,
        "score_time": 0,
    }

    best_model = None

    for i, combination in enumerate(
        tqdm(param_combinations, desc="Hyperparameter Grid Search")
    ):

        current_params = {}  # Current hyperparameters
        for param, value in zip(params.keys(), combination):
            current_params[param] = value  # Update the current hyperparameters

        model = models[model_type](**current_params)

        model.fit(X, y)

        """# Cross validate
        try:
            scores = cross_validate(
                model,
                X,
                y,
                cv=model_config.cv,
                n_jobs=model_config.n_jobs,
                scoring=model_config.scoring
            )
        except Exception as e:
            print(f"Error during cross-validation for run{i}: {str(e)}")
            continue  # Skip this iteration if cross-validation fails"""

        # Predict
        y_pred = model.predict(X_test)
        """y_pred_proba = model.predict_proba(X_test)[
            :, 1
        ]  # Probabilistic predictions for ROC AUC"""

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="micro")
        precision = precision_score(y_test, y_pred, average="micro")
        recall = recall_score(y_test, y_pred, average="micro")
        # roc_auc = roc_auc_score(y_test, y_pred_proba, average='micro', multi_class='ovr')
        cohen_kappa = cohen_kappa_score(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            # "roc_auc": roc_auc,
            "cohen_kappa": cohen_kappa,
            # "mean_test_score": scores["test_score"].mean(),
            # "std_test_score": scores["test_score"].std(),
            # "fit_time": scores["fit_time"].mean(),
            # "score_time": scores["score_time"].mean()
        }

        # Log to MLflow
        log_to_mlflow(
            run_name=model_type + "_" + str(i),
            params=current_params,
            metrics=metrics,
            model=None,
            model_type="xgboost",
            artifacts=None,
            model_name=model_type,
            experiment_name=model_type + "_experiment",
            tracking_uri=config.mlflow.tracking_uri,
        )

        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            # best_params = current_params
            best_model = model

    return best_model


def upsample_dataset(X_train, y_train, training_config, model_type):
    # Upsample the minority class
    X_upsampled = X_train.copy().to_frame()
    X_upsampled[training_config.training[model_type].target] = y_train

    # Separate majority and minority classes
    df_majority = X_upsampled[
        X_upsampled[training_config.training[model_type].target] == 0
    ]
    df_minority = X_upsampled[
        X_upsampled[training_config.training[model_type].target] == 1
    ]

    # Upsample minority class
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=df_majority.shape[0],
        random_state=42,
    )

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Display new class counts
    X_train = df_upsampled[training_config.training[model_type].features]
    y_train = df_upsampled[training_config.training[model_type].target]

    return X_train, y_train
