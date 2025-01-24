import os
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)


class TransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name=None, num_labels=2, local_model_path=None):
        self.model_name = model_name
        self.num_labels = num_labels
        if local_model_path:
            # Load from local path
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                local_model_path, num_labels=num_labels, ignore_mismatched_sizes=True
            )
            self.classes_ = [i for i in range(num_labels)]  # Set classes_ for sklearn compatibility

        elif model_name:
            # Load from Hugging Face hub
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, ignore_mismatched_sizes=True
            )
        else:
            raise ValueError("Either model_name or local_model_path must be provided.")

    def tokenize(self, X):
        return self.tokenizer(
            list(X), truncation=True, padding="max_length", max_length=512, return_token_type_ids=False
        )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
            assert len(encodings["input_ids"]) == len(labels), "Mismatch in lengths"

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    def fit(
        self,
        X,
        y,
        eval_X=None,
        eval_y=None,
        freeze_layers_prefix=None,
        **kwargs,
    ):
        """
        Standard training method on the given data.
        """
        if len(X) != len(y):
            raise ValueError("X and y must be the same length")

        # Tokenize
        encodings = self.tokenize(X)
        train_dataset = self.TextDataset(encodings, y.tolist())

        eval_dataset = None
        if eval_X is not None and eval_y is not None:
            if len(eval_X) != len(eval_y):
                raise ValueError("eval_X and eval_y must be the same length")
            val_encodings = self.tokenize(eval_X)
            eval_dataset = self.TextDataset(val_encodings, eval_y.tolist())

        # Freeze desired layers
        if freeze_layers_prefix:
            for name, param in self.model.named_parameters():
                if any(name.startswith(prefix) for prefix in freeze_layers_prefix):
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        # Define optimizer and scheduler
        lr = kwargs.get("learning_rate", 2e-5)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # Setup Trainer
        training_args = TrainingArguments(
            output_dir=kwargs.get("output_dir", "./results"),
            eval_strategy=kwargs.get("eval_strategy", "epoch" if eval_dataset else "no"),
            save_strategy=kwargs.get("save_strategy", "epoch" if eval_dataset else "no"),
            learning_rate=lr,
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 16),
            per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 16),
            num_train_epochs=kwargs.get("num_train_epochs", 20),
            weight_decay=kwargs.get("weight_decay", 0.01),
            load_best_model_at_end=kwargs.get("load_best_model_at_end", True if eval_dataset else False),
            metric_for_best_model=kwargs.get("metric_for_best_model", "eval_loss"),
            greater_is_better=kwargs.get("greater_is_better", False),
            save_total_limit=kwargs.get("save_total_limit", 2),
            logging_dir=kwargs.get("logging_dir", "./logs"),
            logging_steps=kwargs.get("logging_steps", 10),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.tokenizer, padding="max_length", max_length=512
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=kwargs.get("early_stopping_patience", 2))],
            optimizers=(optimizer, scheduler)
        )

        trainer.train()

        # Update classes_ for sklearn compatibility
        self.classes_ = np.unique(y)
        return self

    def expand_num_labels(self, new_num_labels):
        """
        Create a fresh BERT-based model with 'new_num_labels',
        then copy backbone + partial head weights from the old model.
        """
        print(f"Expanding classifier from {self.num_labels} to {new_num_labels}...")

        if new_num_labels <= self.num_labels:
            raise ValueError("New number of labels must be larger than current num_labels.")

        # Save references to old model & config
        old_model = self.model
        old_num_labels = self.num_labels
        old_config = old_model.config

        # Create a brand-new BERTForSequenceClassification from the same base checkpoint,
        # but with 'new_num_labels' outputs in the classifier.
        # This is needed to ensure that the library transformers handle the new model architecture correctly.
        if self.model_name is not None:
            new_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=new_num_labels,
                ignore_mismatched_sizes=True,
            )
        else:
            # If we loaded from local_model_path, just reuse old_config but override num_labels
            new_model = AutoModelForSequenceClassification.from_pretrained(
                old_config._name_or_path,  # might be a path or the original pretrained name
                num_labels=new_num_labels,
                ignore_mismatched_sizes=True,
            )

        # Copy the entire BERT backbone weights from old -> new to keep the learned weights from last trainings
        new_model.bert.load_state_dict(old_model.bert.state_dict())

        # Partially copy the old classifier's weights into the new classifier.
        # This is important to preserve the learned weights of the old labels.
        with torch.no_grad():
            # Copy old weights/bias for the old_num_labels portion
            new_model.classifier.weight[:old_num_labels] = old_model.classifier.weight
            new_model.classifier.bias[:old_num_labels] = old_model.classifier.bias

            # Initialize the new label region for weights and bias
            torch.nn.init.xavier_uniform_(new_model.classifier.weight[old_num_labels:])
            new_model.classifier.bias[old_num_labels:].zero_()

        # 4. Replace self.model with the new one
        self.model = new_model
        self.num_labels = new_num_labels
        # Also update the config inside our new model
        self.model.config.num_labels = new_num_labels

    def incremental_fit(
            self,
            new_X,
            new_y,
            new_num_labels,
            replay_X=None,
            replay_y=None,
            eval_X=None,
            eval_y=None,
            freeze_layers_prefix=None,
            **kwargs,
    ):
        """
        Incrementally train the model on (new_X, new_y), expanding
        from current num_labels to new_num_labels, plus optional replay data.
        This uses a fresh model approach for the new classification head.
        """
        # 1. Expand classification head if needed
        if new_num_labels > self.num_labels:
            self.expand_num_labels(new_num_labels)

        # 2. Combine new data with replay data
        if replay_X is not None and replay_y is not None:
            X_combined = np.concatenate([new_X, replay_X])
            y_combined = np.concatenate([new_y, replay_y])
        else:
            X_combined = new_X
            y_combined = new_y

        # 3. Fine-tune on combined data
        return self.fit(
            X_combined,
            y_combined,
            eval_X=eval_X,
            eval_y=eval_y,
            freeze_layers_prefix=freeze_layers_prefix,
            **kwargs,
        )

    def predict(self, X):
        encodings = self.tokenize(X)
        dataset = self.TextDataset(encodings, [0] * len(X))  # dummy labels
        trainer = Trainer(model=self.model)
        preds = trainer.predict(dataset)
        logits = preds.predictions
        return np.argmax(logits, axis=1)

    def predict_proba(self, X):
        encodings = self.tokenize(X)
        dataset = self.TextDataset(encodings, [0] * len(X))  # dummy
        trainer = Trainer(model=self.model)
        preds = trainer.predict(dataset)
        logits = preds.predictions
        return F.softmax(torch.from_numpy(logits), dim=1).numpy()

    def save_model(self, save_path):
        """Save the model and tokenizer to the specified path."""
        # Ensure the save_path exists
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}.")

    def get_embeddings(self, X, batch_size=32):
        """
        Get [CLS] token embeddings for the given data in batches.

        Parameters:
            X (iterable): Input texts to process.
            batch_size (int): Number of samples to process per batch.

        Returns:
            np.ndarray: Embeddings for the input texts.
        """
        # Tokenize the input texts
        encodings = self.tokenize(X)

        # Convert tokenized data to PyTorch tensors
        input_ids = torch.tensor(encodings["input_ids"])
        attention_mask = torch.tensor(encodings["attention_mask"])

        # Move tensors to the same device as the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Prepare for batching
        num_samples = input_ids.size(0)
        embeddings = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                # Extract batch
                batch_input_ids = input_ids[i : i + batch_size].to(device)
                batch_attention_mask = attention_mask[i : i + batch_size].to(device)

                # Get the transformer backbone's outputs (hidden states)
                outputs = self.model.bert(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    return_dict=True,
                )
                # Extract the CLS token embedding (first token for each sequence)
                cls_embeddings = outputs.last_hidden_state[
                    :, 0, :
                ]  # Shape: (batch_size, hidden_size)

                # Append the batch embeddings
                embeddings.append(cls_embeddings.cpu().numpy())

        # Concatenate all batch embeddings
        return np.vstack(embeddings)
