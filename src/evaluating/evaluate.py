import argparse
import os
import pandas as pd
import torch
import numpy as np

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.evaluating.utils import plot_confusion_matrix, plot_roc_curves
from src.training.model import TransformerClassifier
from src.utils import load_config, get_logger


def classifier_evaluation(
    env: str,
    input_dataset: str,
    input_model: str,
    output_reports: str,
    model_type: str = "BERT",
):
    """
    Evaluate a trained classifier on a test set and save metrics/reports/plots.
    """

    # Load configs
    config = load_config(file_name="config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)
    logger.info("--------------------------------------------------")
    logger.info(f"Evaluating classifier in {env} environment")

    # Load dataset
    logger.info("Loading dataset...")
    df = pd.read_parquet(input_dataset)

    # Select features and target
    X = df[training_config.training[model_type].features]
    y = df[training_config.training[model_type].target]
    num_labels = y.nunique()

    # Perform train-val-test split exactly as in training
    # (We only need the final test set to evaluate, but we replicate the same approach)
    logger.info("Splitting dataset for evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_config.training[model_type].test_size,
        random_state=training_config.training.random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=training_config.training[model_type].val_size,
        random_state=training_config.training.random_state,
        stratify=y_train,
    )
    X_test, X_cp, y_test, y_cp = train_test_split(
        X_test,
        y_test,
        test_size=training_config.training[model_type].cp_size,
        random_state=training_config.training.random_state,
        stratify=y_test,
    )

    # Load the trained model
    logger.info(f"Loading model from: {input_model}")
    clf = TransformerClassifier(local_model_path=input_model, num_labels=num_labels)

    logger.info("Generating predictions...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)  # Probability estimates

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, digits=4)

    # Compute AUC if binary or multi-class
    try:
        # For binary classification, y_test must be {0,1}, so we handle numeric encoding
        # For multi-class, we do a macro-average
        average_type = "macro" if num_labels > 2 else "binary"
        auc_value = roc_auc_score(
            y_test,
            y_prob[:, 1] if num_labels == 2 else y_prob,
            average=average_type,
            multi_class="ovr",
        )
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC score: {e}")
        auc_value = None

    # Ensure output folder exists
    os.makedirs(output_reports, exist_ok=True)

    # Save classification report
    report_path = os.path.join(output_reports, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Accuracy:\n")
        f.write(f"{acc}\n\n")
        if auc_value is not None:
            f.write("AUC:\n")
            f.write(f"{auc_value}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{clf_report}\n")

    logger.info(f"Evaluation complete. Report saved to {report_path}")

    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        classes=np.unique(y_test),
        title="Confusion Matrix",
        output_path=os.path.join(output_reports, "confusion_matrix.png"),
    )

    # Plot ROC curves and save
    roc_path = os.path.join(output_reports, "roc_curve.png")
    plot_roc_curves(y_test, y_prob, classes=np.unique(y_test), output_path=roc_path)
    logger.info(f"ROC Curve saved to {roc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TransformerClassifier on a test set."
    )
    parser.add_argument(
        "--env", type=str, help="Environment (e.g., dev, prod)", default="dev"
    )
    parser.add_argument(
        "--input-dataset", type=str, help="Path to the input dataset", required=True
    )
    parser.add_argument(
        "--input-model",
        type=str,
        help="Path to the trained model folder",
        required=True,
    )
    parser.add_argument(
        "--output-reports",
        type=str,
        help="Folder to save evaluation reports",
        required=True,
    )

    args = parser.parse_args()

    classifier_evaluation(
        env=args.env,
        input_dataset=args.input_dataset,
        input_model=args.input_model,
        output_reports=args.output_reports,
    )
