import os
from typing import Tuple, List
from mapie._typing import ArrayLike, NDArray

import numpy as np
import pandas as pd
import torch
from mapie.classification import MapieClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def chunked_mapie_predict(
    mapie_clf: MapieClassifier,
    X_test: pd.DataFrame,  # or pd.Series, or np.array
    alpha: float = 0.15,
    chunk_size: int = 64,
):
    """
    Perform MapieClassifier.predict in smaller batches to avoid OOM errors.
    Returns:
      point_preds: shape (n_samples,)
      conf_sets: shape (n_samples, n_alpha, n_classes)
    """
    # If multi-column:
    X_test_list = X_test.values.tolist()

    # If single text column:
    # X_test_list = X_test.squeeze().tolist()

    all_point_preds = []
    all_conf_sets = []

    for start_idx in range(0, len(X_test_list), chunk_size):
        end_idx = start_idx + chunk_size
        X_chunk = X_test_list[start_idx:end_idx]

        with torch.no_grad():
            # p_preds: shape (batch_size,)
            # c_sets: shape (batch_size, n_alpha, n_classes)
            p_preds, c_sets = mapie_clf.predict(X_chunk, alpha=alpha)

        all_point_preds.append(p_preds)
        all_conf_sets.append(c_sets)

    # Concatenate results along axis=0
    point_preds = np.concatenate(all_point_preds, axis=0)
    conf_sets = np.concatenate(all_conf_sets, axis=0)

    # Optional sanity checks
    if conf_sets.shape[0] != len(X_test_list):
        raise ValueError(
            "Mismatch: conf_sets has {}, but X_test_list has {}".format(
                conf_sets.shape[0], len(X_test_list)
            )
        )

    return point_preds, conf_sets


def generate_hist(
    outlier_test: pd.DataFrame, conf_sets: Tuple[NDArray, NDArray], output_reports: str
) -> str:
    """
    Generate a histogram of the distribution of conformal set sizes.
    :param outlier_test:
    :param conf_sets:
    :param output_reports:
    :return:
    """

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
    dist_plot_path = os.path.join(output_reports, "conformal_set_size_distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()

    return dist_plot_path


def generate_barplot(
    outliers_df: pd.DataFrame, training_config, model_type: str, output_reports: str
) -> str:
    """
    Generate a bar plot of the number of outliers by true label.
    :param outliers_df:
    :param training_config:
    :param model_type:
    :param output_reports:
    :return:
    """
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

    return outliers_bar_path


def generate_classification_report(
    outlier_test: pd.DataFrame, output_reports: str
) -> str:
    # Convert booleans => int
    y_true = outlier_test["true_outlier"].astype(int).values
    y_pred = outlier_test["outlier"].astype(int).values

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cls_report = classification_report(y_true, y_pred, zero_division=0)

    # Log to a text file
    eval_report_file = os.path.join(output_reports, "outlier_evaluation_report.txt")
    with open(eval_report_file, "w", encoding="utf-8") as f:
        f.write("Outlier Detection: Binary Classification Evaluation\n")
        f.write("--------------------------------------------------\n\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Full Classification Report:\n")
        f.write(f"{cls_report}\n")

    return eval_report_file


def generate_confusion_matrix(outlier_test: pd.DataFrame, output_reports: str):
    # Convert booleans => int
    y_true = outlier_test["true_outlier"].astype(int).values
    y_pred = outlier_test["outlier"].astype(int).values

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Outlier Detection Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ["Normal (0)", "Outlier (1)"], rotation=45, ha="right")
    plt.yticks(ticks, ["Normal (0)", "Outlier (1)"])
    # Print counts inside squares
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    cm_path = os.path.join(output_reports, "outlier_detection_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    return cm_path


def generate_plot_scores(n, alphas, scores, quantiles, output_reports):
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    plt.figure(figsize=(7, 5))
    plt.hist(scores, bins="auto")
    for i, quantile in enumerate(quantiles):
        plt.vlines(
            x=quantile,
            ymin=0,
            ymax=100,
            color=colors[i],
            ls="dashed",
            label=f"alpha = {alphas[i]}",
        )
    plt.title("Distribution of scores")
    plt.legend()
    plt.xlabel("Scores")
    plt.ylabel("Count")

    scores_plot_path = os.path.join(output_reports, "scores_distribution.png")
    plt.savefig(scores_plot_path)
    plt.close()

    return scores_plot_path


def plot_prediction_decision(
    X_train: pd.DataFrame, y_train, X_test: pd.DataFrame, y_pred_mapie, ax
):
    colors = {0: "#1f77b4", 1: "#ff7f0e"}
    y_train_col = list(map(colors.get, y_train))
    y_pred_col = list(map(colors.get, y_pred_mapie))

    ax.scatter(
        X_test["dim_0"],
        X_test["dim_1"],
        color=y_pred_col,
        marker=".",
        s=10,
        alpha=0.4,
    )
    ax.scatter(
        X_train["dim_0"],
        X_train["dim_1"],
        color=y_train_col,
        marker="o",
        s=10,
        edgecolor="k",
    )
    ax.set_title("Predicted labels")


def plot_prediction_set(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_ps: NDArray,
    alpha_: float,
    ax,
) -> None:
    colors = {0: "#1f77b4", 1: "#ff7f0e"}
    y_train_col = list(map(colors.get, y_train))

    tab10 = plt.cm.get_cmap("Purples", 4)
    y_pi_sums = y_ps.sum(axis=1)
    num_labels = ax.scatter(
        X_test["dim_0"],
        X_test["dim_1"],
        c=y_pi_sums,
        marker="o",
        s=10,
        alpha=1,
        cmap=tab10,
        vmin=0,
        vmax=3,
    )
    ax.scatter(
        X_train["dim_0"],
        X_train["dim_1"],
        color=y_train_col,
        marker="o",
        s=10,
        edgecolor="k",
    )
    ax.set_title(f"Number of labels for alpha={alpha_}")
    plt.colorbar(num_labels, ax=ax)


def generate_cp_results(
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    alphas: List[float],
    y_pred_mapie: NDArray,
    y_ps_mapie: NDArray,
    output_reports: str,
) -> str:
    _, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(10, 10))
    axs = {0: ax1, 1: ax2, 2: ax3, 3: ax4}
    plot_prediction_decision(X_train, y_train, X_test, y_pred_mapie, axs[0])
    for i, alpha_ in enumerate(alphas):
        plot_prediction_set(
            X_train, y_train, X_test, y_ps_mapie[:, :, i], alpha_, axs[i + 1]
        )
    results_plot_path = os.path.join(output_reports, "cp_results.png")
    plt.savefig(results_plot_path)
    plt.close()
    return results_plot_path


def reduce_dimensions(
    embeddings: pd.DataFrame,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reduce the dimensionality of embeddings to 2D for visualization.

    Parameters:
        embeddings (pd.DataFrame or np.ndarray): High-dimensional data.
        method (str): The dimensionality reduction method. Options: 'pca', 'tsne'.
        n_components (int): Number of dimensions to reduce to (default: 2).
        random_state (int): Random state for reproducibility.

    Returns:
        pd.DataFrame: Reduced 2D embeddings as a DataFrame with columns ['dim_0', 'dim_1'].
    """
    """if not isinstance(embeddings, (pd.DataFrame, np.ndarray)):
        raise ValueError("Input embeddings must be a pandas DataFrame or numpy array.")"""

    if isinstance(embeddings, pd.DataFrame):
        embeddings = embeddings.values  # Convert to numpy array if DataFrame

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state)
    else:
        raise ValueError("Unsupported method. Choose 'pca' or 'tsne'.")

    reduced = reducer.fit_transform(embeddings)
    return pd.DataFrame(reduced, columns=[f"dim_{i}" for i in range(n_components)])
