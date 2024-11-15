from typing import Any, Optional

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import mlflow


def custom_grid_search(
        pipeline: Pipeline,
        parameters: dict,
        X: Any,
        scoring: Any = None,
        cv: int = 3,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42
):

    grid_search = GridSearchCV(
        pipeline, parameters, cv=cv, n_jobs=n_jobs, verbose=verbose
    )

    grid_search.fit(X)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_

    return best_params, best_score, best_estimator


def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    score = silhouette_score(X, labels)
    return score



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

            # Register the model in the Model Registry with a specific name
            model_uri = f"runs:/{run.info.run_id}/{model_type}"
            model_version = mlflow.register_model(model_uri, name=model_name)

            print(f"Model {model_version.name} registered with version: {model_version.version}")

        if artifacts:
            for key, value in artifacts.items():
                mlflow.log_artifact(value, key)