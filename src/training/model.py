import mlflow
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
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
                local_model_path, num_labels=num_labels
            )
            self.classes_ = [i for i in range(num_labels)]  # Set classes_
            # self.n_features_in_ = 512  # Set n_features_in_ (optional, for compatibility)

        elif model_name:
            # Load from Hugging Face hub
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )
        else:
            raise ValueError("Either model_name or local_model_path must be provided.")

    def tokenize(self, X):
        return self.tokenizer(
            list(X), truncation=True, padding="max_length", max_length=512
        )

    def fit(self, X, y, eval_X=None, eval_y=None, freeze_layers_prefix: list = None):
        # Tokenize the input texts
        encodings = self.tokenize(X)

        # Dataset Class
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

        train_dataset = Dataset(encodings, y.tolist())
        eval_dataset = None
        if eval_X is not None and eval_y is not None:
            eval_encodings = self.tokenize(eval_X)
            eval_dataset = Dataset(eval_encodings, eval_y.tolist())

        # Freeze layers
        if freeze_layers_prefix:
            for name, param in self.model.named_parameters():
                if not any(name.startswith(prefix) for prefix in freeze_layers_prefix):
                    param.requires_grad = False

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch" if eval_dataset is not None else "no",
            learning_rate=2e-5,
            lr_scheduler_type="reduce_lr_on_plateau",  # Scheduler to adjust learning rate
            per_device_train_batch_size=16,
            num_train_epochs=20,
            weight_decay=0.01,
            save_total_limit=2,
            logging_dir="./logs",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.tokenizer, padding="max_length", max_length=512
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        # Train the model
        trainer.train()

        # Set the classes_ attribute required by scikit-learn
        self.classes_ = torch.unique(torch.tensor(y.to_list())).numpy()

        return self

    def predict(self, X):
        encodings = self.tokenize(X)

        # Convert tokenized data to PyTorch tensors
        input_ids = torch.tensor(encodings["input_ids"])
        attention_mask = torch.tensor(encodings["attention_mask"])

        # Move tensors to the same device as the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X):
        encodings = self.tokenize(X)

        # Convert tokenized data to PyTorch tensors
        input_ids = torch.tensor(encodings["input_ids"])
        attention_mask = torch.tensor(encodings["attention_mask"])

        # Move tensors to the same device as the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer saved to {save_path}.")
