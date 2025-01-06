import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import matplotlib

matplotlib.use("Agg")  # For headless environments (comment out if running locally)
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(
    cm,
    classes,
    output_path,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    """
    Print and plot the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(output_path)
    plt.close()


def plot_roc_curves(y_true, y_probs, classes, output_path):
    """
    Plot ROC curves (one-vs-rest if multi-class) and save to file.
    """

    # Binarize the labels (important for multi-class)
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    # If binary classification, handle it in a simpler way
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.close()
    else:
        # For multi-class, do a one-vs-rest approach
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})"
                "".format(classes[i], roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "r--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Some extension of ROC to multi-class")
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.close()
