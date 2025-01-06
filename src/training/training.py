import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import torch

from src.training.model import TransformerClassifier
from src.training.utils import upsample_dataset
from src.utils import load_config, get_logger


def classifier_training(
    env: str, input_dataset: str, output_model: str, model_type: str = "BERT"
):

    config = load_config(file_name="config", env=env)
    processing_config = load_config(file_name="processing_config", env=env)
    training_config = load_config(file_name="training_config", env=env)
    logger = get_logger(config)

    logger.info(f"--------------------------------------------------")
    logger.info(f"Start training the classifier in {env} environment")

    # Load dataset
    logger.info("Loading dataset...")
    df = pd.read_parquet(input_dataset)
    X = df[["processed_text_to_analyse", "label"]]
    num_labels = X["label"].nunique()

    # Split dataset into train and test sets
    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        # X.drop(columns=[training_config.training[model_type].target]),
        X[training_config.training[model_type].features],
        X[training_config.training[model_type].target],
        test_size=training_config.training[model_type].test_size,
        random_state=training_config.training.random_state,
        stratify=X[training_config.training[model_type].target],
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

    # Upsample the minority class
    logger.info(f"Upsample the minority class")
    X_train, y_train = upsample_dataset(X_train, y_train, training_config, model_type)

    # Initialize the transformer classifier
    clf = TransformerClassifier(
        model_name="dtorber/bert-base-spanish-wwm-cased_K4",
        num_labels=num_labels,
    )

    # Train the model
    logger.info("Training TransformerClassifier...")
    clf.fit(
        X=X_train,
        y=y_train,
        eval_X=X_val,
        eval_y=y_val,
        freeze_layers_prefix=["bert.embeddings", "bert.encoder"],
    )

    # Save the trained model
    print("Saving the trained model...")
    clf.model.save_pretrained(output_model)
    clf.tokenizer.save_pretrained(output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a BERT model for text classification"
    )
    parser.add_argument(
        "--env", type=str, help="Environment (e.g., dev, prod)", default="dev"
    )
    parser.add_argument(
        "--input-dataset", type=str, help="Path to the input dataset", required=True
    )
    parser.add_argument(
        "--output-model", type=str, help="Path to save the output model", required=True
    )
    parser.add_argument("--num-labels", type=int, help="Number of labels", default=4)

    args = parser.parse_args()

    classifier_training(
        env=args.env,
        input_dataset=args.input_dataset,
        output_model=args.output_model,
    )
