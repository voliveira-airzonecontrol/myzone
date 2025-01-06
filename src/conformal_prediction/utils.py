import numpy as np
import pandas as pd
import torch
from mapie.classification import MapieClassifier

def chunked_mapie_predict(
    mapie_clf: MapieClassifier,
    X_test: pd.DataFrame,  # or pd.Series, or np.array
    alpha: float = 0.15,
    chunk_size: int = 64
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
        raise ValueError("Mismatch: conf_sets has {}, but X_test_list has {}".format(
            conf_sets.shape[0], len(X_test_list)
        ))

    return point_preds, conf_sets
