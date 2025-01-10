import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

from mapie.classification import MapieClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.conformal_prediction.utils import (
    chunked_mapie_predict,
    generate_hist,
    generate_barplot,
    generate_classification_report,
    generate_confusion_matrix,
    generate_plot_scores,
    generate_cp_results,
    reduce_dimensions,
)

# For loading your trained TransformerClassifier
from src.training.model import TransformerClassifier
from src.utils import load_config, get_logger


def detect_outliers(
    env: str,
    input_dataset: str,
    input_model: str,
    input_outliers: str,
    output_reports: str,
    model_type: str = "BERT",
) -> None:
    """
    Use MAPIE to generate conformal prediction sets on the given dataset
    and label outliers where the prediction set is larger than 1 at alpha=0.15.
    """
    config = load_config(file_name="config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)
    logger.info("--------------------------------------------------")
    logger.info(f"Running outlier detection in {env} environment.")

    # Load the dataset
    logger.info(f"Loading dataset: {input_dataset}")
    df = pd.read_parquet(input_dataset)

    # Select features and target
    X = df[training_config.training[model_type].features]
    y = df[training_config.training[model_type].target]
    num_labels = y.nunique()

    # -----------------------------------------------------------------------------
    # Perform train-val-test split exactly as in training
    # (We only need the final test set to evaluate, but we replicate the same approach)
    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
    # Load outliers and concat with X_test and y_test
    # -----------------------------------------------------------------------------
    logger.info(f"Loading outliers from: {input_outliers}")
    outliers = pd.read_parquet(input_outliers)
    X_outliers = outliers[training_config.training[model_type].features]
    y_outliers = outliers[training_config.training[model_type].target]
    X_test = pd.concat([X_test, X_outliers], ignore_index=True)
    y_test = pd.concat([y_test, y_outliers], ignore_index=True)

    # -----------------------------------------------------------------------------
    # Load the trained model
    # -----------------------------------------------------------------------------
    logger.info(f"Loading pre-trained model from: {input_model}")
    clf = TransformerClassifier(local_model_path=input_model, num_labels=num_labels)

    # -----------------------------------------------------------------------------
    # Prepare the model for conformal prediction
    # -----------------------------------------------------------------------------
    logger.info("Calibrating model for conformal prediction...")
    mapie_clf = MapieClassifier(
        estimator=clf,
        method="score",
        cv="prefit",
        random_state=training_config.training.random_state,
    )
    # Fit on the calibration set
    mapie_clf.fit(X_cp, y_cp)

    alpha = training_config.training["conformal_prediction"].alpha

    logger.info(f"Generating conformal prediction sets with alpha={alpha}")
    # point_preds, conf_sets = mapie_clf.predict(X_test, alpha=alpha)
    point_preds, conf_sets = chunked_mapie_predict(mapie_clf, X_test, alpha=alpha)

    # -----------------------------------------------------------------------------
    # Detect outliers
    # -----------------------------------------------------------------------------
    logger.info("Detecting outliers...")
    outlier_test = pd.concat([X_test, y_test], axis=1)
    outlier_test["true_outlier"] = y_test.apply(lambda x: 0 if x in [0, 1] else 1)
    outlier_test["outlier"] = False
    if conf_sets is not None:
        for i in range(outlier_test.shape[0]):
            label_boolean = conf_sets[
                i, :, 0
            ]  # The first value of Alpha is used to classify outliers
            set_size = np.sum(label_boolean)
            # If the set size is larger than 1 or equal to 0, we label it as an outlier
            if set_size != 1:
                outlier_test.loc[outlier_test.index[i], "outlier"] = True

    # -----------------------------------------------------------------------------
    # Save outliers to a parquet file
    # -----------------------------------------------------------------------------
    outliers_df = outlier_test[outlier_test["outlier"]]
    logger.info(f"Number of outliers detected: {len(outliers_df)}")
    os.makedirs(output_reports, exist_ok=True)
    outliers_file = os.path.join(output_reports, "detected_outliers.parquet")
    outliers_df.to_parquet(outliers_file, index=False)
    logger.info(f"Outliers saved to: {outliers_file}")

    # -----------------------------------------------------------------------------
    # Distribution of scores
    # -----------------------------------------------------------------------------
    if conf_sets is not None:
        logger.info("Generating distribution of scores plot.")
        conformity_scores = mapie_clf.conformity_scores_
        quantiles = mapie_clf.conformity_score_function_.quantiles_
        n = len(mapie_clf.conformity_scores_)
        scores_plot_path = generate_plot_scores(
            n, alpha, conformity_scores, quantiles, output_reports
        )
        logger.info(f"Distribution of scores plot saved to {scores_plot_path}")

    # -----------------------------------------------------------------------------
    # Conformal prediction results
    # -----------------------------------------------------------------------------
    """if conf_sets is not None:
        logger.info("Saving conformal prediction results.")
        X_train_reduced = reduce_dimensions(X_cp[training_config.training[model_type].features], method="pca")
        X_test_reduced = reduce_dimensions(outlier_test[training_config.training[model_type].features], method="pca")

        results_plot_path = generate_cp_results(
            X_train=X_train_reduced,
            y_train=y_cp,
            X_test=X_test_reduced,
            alphas=alpha,
            y_pred_mapie=point_preds,
            y_ps_mapie=conf_sets,
            output_reports=output_reports,
        )
        logger.info(f"Conformal prediction results saved to: {results_plot_path}")"""

    # -----------------------------------------------------------------------------
    # Distribution of set sizes (for those who want a deeper analysis)
    # -----------------------------------------------------------------------------
    if conf_sets is not None:
        logger.info("Generating distribution of conformal set sizes.")
        dist_plot_path = generate_hist(outlier_test, conf_sets, output_reports)
        logger.info(f"Distribution of set sizes plot saved to {dist_plot_path}")

    # -----------------------------------------------------------------------------
    # Bar chart: number of outliers per (predicted) label
    # -----------------------------------------------------------------------------
    if outliers_df.shape[0] > 0:
        logger.info("Generating outliers by class plot.")
        outliers_bar_path = generate_barplot(
            outliers_df, training_config, model_type, output_reports
        )
        logger.info(f"Outliers by class plot saved to {outliers_bar_path}")

    # -----------------------------------------------------------------------------
    # EVALUATE OUTLIER DETECTION AS BINARY CLASSIFICATION
    # -----------------------------------------------------------------------------
    if "true_outlier" in outlier_test.columns:

        # Generate classification report
        eval_report_file = generate_classification_report(outlier_test, output_reports)
        logger.info(f"Outlier evaluation report saved to {eval_report_file}")

        # Generate Confusion Matrix
        cm_path = generate_confusion_matrix(outlier_test, output_reports)
        logger.info(f"Outlier detection confusion matrix saved to {cm_path}")

    else:
        logger.warning(
            "No 'true_outlier' column found in data. Skipping binary classification evaluation."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Outlier detection using MAPIE conformal sets."
    )
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev, prod, etc.)"
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        required=True,
        help="Path to test+unseen dataset parquet file.",
    )
    parser.add_argument(
        "--input-model",
        type=str,
        required=True,
        help="Path to the trained model folder.",
    )
    parser.add_argument(
        "--input-outliers",
        type=str,
        required=True,
        help="Path to the outliers parquet file.",
    )
    parser.add_argument(
        "--output-reports",
        type=str,
        required=True,
        help="Folder to store outliers and reports.",
    )

    args = parser.parse_args()

    detect_outliers(
        env=args.env,
        input_dataset=args.input_dataset,
        input_model=args.input_model,
        input_outliers=args.input_outliers,
        output_reports=args.output_reports,
    )
