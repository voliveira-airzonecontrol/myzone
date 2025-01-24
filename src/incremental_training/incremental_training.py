import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch

from src.training.model import TransformerClassifier
from src.training.utils import upsample_dataset
from src.utils import load_config, get_logger
from src.training import model


def incremental_training(
        env: str,
        input_dataset: str,
        input_new_class: str,
        input_true_new_class: str,
        input_model: str,
        output_model: str,
        model_type: str = "BERT"
) -> None:

    config = load_config(file_name="config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start incremental training in {env} environment")

    # Load datasets
    logger.info("Loading datasets...")
    known_classes = pd.read_parquet(input_dataset)
    new_class = pd.read_parquet(input_new_class)
    true_new_class = pd.read_parquet(input_true_new_class)

    features = training_config.training[model_type].features
    target = training_config.training[model_type].target

    known_classes = known_classes[[features, target]]
    new_class = new_class[[features, target]]
    true_new_class = true_new_class[[features, target]]

    num_labels = known_classes[target].nunique()

    # Split the dataset into train and test sets
    logger.info("Splitting dataset...")
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

    # Upsample
    logger.info(f"Upsample the minority class")
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

    # Concatenate the original and new class
    logger.info("Concatenating the original and new class...")
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

    # Create test set
    # Remove the detected new class from the true new class dataset (avoid data leakage)
    true_new_class = true_new_class[~true_new_class[features].isin(df_train[features])]

    # Concatenate the original test set and the true new class
    original_df = pd.DataFrame({features: original_X_test.values, target: original_y_test.values})
    df_test = pd.concat([original_df, true_new_class])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=training_config.training.random_state, stratify=y
    )

    X_train, y_train = upsample_dataset(X_train, y_train, training_config, model_type)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # Train the transformer classifier
    logger.info("Training the transformer classifier...")
    new_num_labels = len(y_train.unique())

    clf = model.TransformerClassifier(
        model_name=None, num_labels=num_labels,
        local_model_path=input_model,
    )

    clf_incremented = clf.incremental_fit(
        X_train,
        y_train,
        new_num_labels,
        eval_X=X_val,
        eval_y=y_val,
        learning_rate=2e-5,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        early_stopping_patience=2,
        # freeze_layers_prefix=["bert.embeddings", "bert.encoder"],
    )

    # Save the trained model
    logger.info("Saving the trained model...")
    clf_incremented.save_model(output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Incremental training of a classifier"
    )
    parser.add_argument(
        "--env", type=str, help="Environment (e.g., dev, prod)", default="dev"
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        help="Path to the input dataset",
        required=True,
    )
    parser.add_argument(
        "--input-new-class",
        type=str,
        help="Path to the input new detected class",
        required=True,
    )
    parser.add_argument(
        "--input-true-new-class",
        type=str,
        help="Path to the input true new class",
        required=True,
    )
    parser.add_argument(
        "--input-model",
        type=str,
        help="Path to the input model",
        required=True,
    )
    parser.add_argument(
        "--output-model",
        type=str,
        help="Path to save the output model",
        required=True,
    )

    args = parser.parse_args()

    incremental_training(
        env=args.env,
        input_dataset=args.input_dataset,
        input_new_class=args.input_new_class,
        input_true_new_class=args.input_true_new_class,
        input_model=args.input_model,
        output_model=args.output_model,
    )
