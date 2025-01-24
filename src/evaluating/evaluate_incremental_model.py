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
from sklearn.utils import resample

from src.evaluating.utils import plot_confusion_matrix, plot_roc_curves
from src.training.model import TransformerClassifier
from src.utils import load_config, get_logger


def incremented_classifier_evaluation(
        env: str,
        input_dataset: str,
        input_new_class: str,
        input_true_new_class: str,
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
    known_classes = pd.read_parquet(input_dataset)
    new_class = pd.read_parquet(input_new_class)
    true_new_class = pd.read_parquet(input_true_new_class)

    # Keep only identified new classes
    new_class = new_class[new_class["new_class"]]

    features = training_config.training[model_type].features
    target = training_config.training[model_type].target

    # Select features and target
    known_classes = known_classes[[features, target]]
    new_class = new_class[[features, target]]
    true_new_class = true_new_class[[features, target]]
    num_labels = known_classes[target].nunique()

    # -----------------------------------------------------------------------------------------------------------------
    # Redo train-val-test split exactly as in training
    # -----------------------------------------------------------------------------------------------------------------

    # Perform train-val-test split exactly as in incremental training
    # (We only need the final test set to evaluate, but we replicate the same approach)
    logger.info("Splitting dataset for evaluation...")

    X = known_classes

    original_X_train, original_X_test, original_y_train, original_y_test = train_test_split(
        # X.drop(columns=[training_config.training[model_type].target]),
        X[training_config.training[model_type].features],
        X[training_config.training[model_type].target],
        test_size=training_config.training[model_type].test_size,
        random_state=training_config.training.random_state,
        stratify=X[training_config.training[model_type].target],
    )
    original_X_test, original_X_cp, original_y_test, original_y_cp = train_test_split(
        original_X_test,
        original_y_test,
        test_size=training_config.training[model_type].cp_size,
        random_state=training_config.training.random_state,
        stratify=original_y_test,
    )

    # Upsample the minority class
    X_upsampled = original_X_train.copy().to_frame()
    X_upsampled[training_config.training[model_type].target] = original_y_train

    # Separate majority and minority classes
    df_majority = X_upsampled[X_upsampled[training_config.training[model_type].target] == 0]
    df_minority = X_upsampled[X_upsampled[training_config.training[model_type].target] == 1]

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
    original_X_train = df_upsampled[training_config.training[model_type].features]
    original_y_train = df_upsampled[training_config.training[model_type].target]
    df_original_train = pd.concat([original_X_train, original_y_train], axis=1)

    df_train = pd.concat(
        [
            df_original_train.sample(
                frac=0.1, random_state=training_config.training.random_state
            ),
            new_class,
        ]
    )
    X = df_train[features]
    y = df_train[target]

    # -----------------------------------------------------------------------------------------------------------------
    # Prepare test set
    # -----------------------------------------------------------------------------------------------------------------

    # Remove the detected new class from the true new class dataset (avoid data leakage)
    true_new_class = true_new_class[~true_new_class[features].isin(df_train[features])]

    # Concatenate the original test set and the true new class
    original_df = pd.DataFrame({features: original_X_test.values, target: original_y_test.values})
    df_test = pd.concat([original_df, true_new_class])

    # -----------------------------------------------------------------------------------------------------------------
    # Load the trained model
    # -----------------------------------------------------------------------------------------------------------------
    logger.info(f"Loading model from: {input_model}")
    new_num_labels = len(y.unique())
    clf_incremented = TransformerClassifier(
        model_name=None, num_labels=new_num_labels,
        local_model_path=input_model,
    )

    # -----------------------------------------------------------------------------------------------------------------
    # Generate predictions and evaluate
    # -----------------------------------------------------------------------------------------------------------------
    logger.info("Generating predictions...")
    y_pred = clf_incremented.predict(df_test[features])
    y_prob = clf_incremented.predict_proba(df_test[features])  # Probability estimates

    # Calculate metrics
    acc = accuracy_score(df_test[target], y_pred)
    cm = confusion_matrix(df_test[target], y_pred)
    clf_report = classification_report(df_test[target], y_pred, digits=4)

    # -----------------------------------------------------------------------------------------------------------------
    # Compute AUC if binary or multi-class
    # -----------------------------------------------------------------------------------------------------------------
    try:
        # For binary classification, y_test must be {0,1}, so we handle numeric encoding
        # For multi-class, we do a macro-average
        average_type = "macro" if num_labels > 2 else None
        auc_value = roc_auc_score(
            df_test[target],
            y_prob[:, 1] if num_labels == 2 else y_prob,
            average=average_type,
            multi_class="ovr",
        )
    except Exception as e:
        logger.warning(f"Could not compute ROC-AUC score: {e}")
        auc_value = None

    # Ensure output folder exists
    os.makedirs(output_reports, exist_ok=True)

    # -----------------------------------------------------------------------------------------------------------------
    # Save classification report
    # -----------------------------------------------------------------------------------------------------------------
    report_path = os.path.join(output_reports, "incremental_classification_report.txt")
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

    # -----------------------------------------------------------------------------------------------------------------
    # Plot confusion matrix
    # -----------------------------------------------------------------------------------------------------------------
    plot_confusion_matrix(
        cm,
        classes=np.unique(df_test[target]),
        title="Confusion Matrix",
        output_path=os.path.join(output_reports, "incremental_confusion_matrix.png"),
    )

    # -----------------------------------------------------------------------------------------------------------------
    # Plot ROC curves and save
    # -----------------------------------------------------------------------------------------------------------------
    roc_path = os.path.join(output_reports, "incremental_roc_curve.png")
    plot_roc_curves(df_test[target], y_prob, classes=np.unique(df_test[target]), output_path=roc_path)
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
        "--input-new-class",
        type=str,
        help="Path to the new class dataset",
        required=True,
    )
    parser.add_argument(
        "--input-true-new-class",
        type=str,
        help="Path to the true new class dataset",
        required=True,
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

    incremented_classifier_evaluation(
        env=args.env,
        input_dataset=args.input_dataset,
        input_new_class=args.input_new_class,
        input_true_new_class=args.input_true_new_class,
        input_model=args.input_model,
        output_reports=args.output_reports,
    )
