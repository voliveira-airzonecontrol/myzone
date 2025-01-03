import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline,
)
import torch


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name, num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def fit(self, X, y):
        # Tokenize the input texts
        encodings = self.tokenizer(
            list(X), truncation=True, padding=True, max_length=512
        )

        # Convert to PyTorch dataset
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {
                    key: torch.tensor(val[idx]) for key, val in self.encodings.items()
                }
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        dataset = Dataset(encodings, y.tolist())

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=10,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )

        # Train the model
        trainer.train()

        return self

    def predict(self, X):
        # Tokenize the input texts
        encodings = self.tokenizer(
            list(X), truncation=True, padding=True, max_length=512, return_tensors="pt"
        )

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
        logits = outputs.logits
        return logits.argmax(dim=1).numpy()

    def predict_proba(self, X):
        # Tokenize the input texts
        encodings = self.tokenizer(
            list(X), truncation=True, padding=True, max_length=512, return_tensors="pt"
        )

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**encodings)
        logits = outputs.logits
        return torch.nn.functional.softmax(logits, dim=1).numpy()


def classifier_training(
    env: str, input_dataset: str, output_model: str, num_labels: int = 4
):
    # Load dataset
    print("Loading dataset...")
    df = pd.read_parquet(input_dataset)

    # Split dataset into train and test sets
    print("Splitting dataset...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42
    )

    # Initialize the transformer classifier
    print("Initializing TransformerClassifier...")
    transformer_clf = TransformerClassifier(
        model_name="dtorber/bert-base-spanish-wwm-cased_K4", num_labels=num_labels
    )

    # Train the classifier
    print("Training TransformerClassifier...")
    transformer_clf.fit(train_texts, train_labels)

    # Save the trained model
    print("Saving the trained model...")
    torch.save(transformer_clf.model.state_dict(), output_model)

    # Evaluate the model
    print("Evaluating TransformerClassifier...")
    predictions = transformer_clf.predict(val_texts)
    print("Predictions:", predictions)


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
        num_labels=args.num_labels,
    )
