import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt

from mapie.classification import MapieClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from src.conformal_prediction.utils import chunked_mapie_predict
# For loading your trained TransformerClassifier
from src.training.model import TransformerClassifier
from src.utils import load_config, get_logger


def detect_outliers(
    env: str,
    input_dataset: str,
    input_model: str,
    input_outliers: str,
    output_reports: str,
    alpha: float = 0.15,
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

    # Load outliers and concat with X_test and y_test
    logger.info(f"Loading outliers from: {input_outliers}")
    outliers = pd.read_parquet(input_outliers)
    X_outliers = outliers[training_config.training[model_type].features]
    y_outliers = outliers[training_config.training[model_type].target]
    X_test = pd.concat([X_test, X_outliers], ignore_index=True)
    y_test = pd.concat([y_test, y_outliers], ignore_index=True)

    # Sample X_test and y_test
    # X_test = X_test.sample(frac=0.05, random_state=training_config.training.random_state).reset_index(drop=True)
    # y_test = y_test.sample(frac=0.05, random_state=training_config.training.random_state).reset_index(drop=True)

    # Load the trained model
    logger.info(f"Loading pre-trained model from: {input_model}")
    clf = TransformerClassifier(local_model_path=input_model, num_labels=num_labels)

    # Prepare the model for conformal prediction
    logger.info("Calibrating model for conformal prediction...")
    mapie_clf = MapieClassifier(
        estimator=clf,
        method="score",
        cv="prefit",
        random_state=training_config.training.random_state,
    )
    # Fit on the calibration set
    mapie_clf.fit(X_cp, y_cp)

    logger.info("Generating conformal prediction sets with alpha=0.15")
    # point_preds, conf_sets = mapie_clf.predict(X_test, alpha=alpha)
    point_preds, conf_sets = chunked_mapie_predict(mapie_clf, X_test, alpha=alpha)

    logger.info("Detecting outliers...")
    outlier_test = pd.concat([X_test, y_test], axis=1)
    outlier_test["outlier"] = False
    if conf_sets is not None:
        for i in range(outlier_test.shape[0]):
            label_boolean = conf_sets[i, 0, :]
            set_size = np.sum(label_boolean)
            if set_size > 1:
                outlier_test.loc[outlier_test.index[i], "outlier"] = True

    # Save outliers to a parquet file
    outliers_df = outlier_test[outlier_test["outlier"]]
    logger.info(f"Number of outliers detected: {len(outliers_df)}")
    os.makedirs(output_reports, exist_ok=True)
    outliers_file = os.path.join(output_reports, "detected_outliers.parquet")
    outliers_df.to_parquet(outliers_file, index=False)
    logger.info(f"Outliers saved to: {outliers_file}")

    # 9) Create some analytical plots:
    #    a) Distribution of set sizes (for those who want a deeper analysis)
    if conf_sets is not None:
        logger.info("Generating distribution of conformal set sizes.")
        set_sizes = []
        for i in range(outlier_test.shape[0]):
            if outlier_test["outlier"].iloc[i] is not None:
                # measure how big the set is
                label_boolean = conf_sets[i, :, 0]
                set_size = np.sum(label_boolean)
                set_sizes.append(set_size)

        plt.figure(figsize=(8, 6))
        plt.hist(
            set_sizes,
            bins=range(1, max(set_sizes) + 2),
            color="skyblue",
            edgecolor="black",
        )
        plt.title("Distribution of Conformal Set Sizes (alpha=0.15)")
        plt.xlabel("Set size")
        plt.ylabel("Frequency")
        plt.xticks(range(1, max(set_sizes) + 2))
        dist_plot_path = os.path.join(
            output_reports, "conformal_set_size_distribution.png"
        )
        plt.savefig(dist_plot_path)
        plt.close()
        logger.info(f"Distribution of set sizes plot saved to {dist_plot_path}")

    #    b) Bar chart: number of outliers per (predicted) label
    #       If we have actual labels or predicted labels, we can show how outliers are distributed.
    #       For demonstration, let's show how many outliers exist for each *ground-truth* label.
    if outliers_df.shape[0] > 0:
        outliers_by_class = (
            outliers_df.groupby(training_config.training[model_type].target)
            .size()
            .sort_values(ascending=False)
        )
        plt.figure(figsize=(10, 6))
        outliers_by_class.plot(kind="bar", color="tomato")
        plt.title("Number of Outliers by True Label")
        plt.xlabel("Label")
        plt.ylabel("Count of Outliers")
        outliers_bar_path = os.path.join(output_reports, "outliers_by_class.png")
        plt.savefig(outliers_bar_path)
        plt.close()
        logger.info(f"Outliers by class plot saved to {outliers_bar_path}")


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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.15,
        help="Alpha for conformal prediction sets. Default=0.15",
    )

    args = parser.parse_args()

    detect_outliers(
        env=args.env,
        input_dataset=args.input_dataset,
        input_model=args.input_model,
        input_outliers=args.input_outliers,
        output_reports=args.output_reports,
        alpha=args.alpha,
    )
