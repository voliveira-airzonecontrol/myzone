import numpy as np
import pandas as pd
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import string


class TfIdfPreprocessor:
    """
    Class for encode text data using the TF-IDF method
    This class is intended to be used as a transformer in the sklearn pipeline
    """

    def __init__(
            self,
            min_df=0.01,
            max_df=0.99,
    ):
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None

    def fit(self, X, y=None):
        """
        Fit the vectorizer to the text data
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self.vectorizer.fit(X)
        return self

    def transform(self, X, y=None):
        """
        Transform the text data using the fitted vectorizer
        """
        return self.vectorizer.transform(X)

    def get_feature_names(self):
        """
        Get feature names from the vectorizer
        """
        return self.vectorizer.get_feature_names_out()

    def get_params(self, deep=True):
        """
        Get parameters for this estimator
        """
        return {
            "min_df": self.min_df,
            "max_df": self.max_df,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class Doc2VecPreprocessor:
    """
    Class for encode text data using the Doc2Vec method
    This class is intended to be used as a transformer in the sklearn pipeline
    """

    def __init__(
            self,
            dm=0,
            vector_size=100,
            epochs=10,
            min_count=1,
            workers=4,
            sample=0,
            negative=5,
            seed=0,
            hs=0,
            custom_tokenizer=None,
    ):
        self.dm = dm
        self.vector_size = vector_size
        self.epochs = epochs
        self.min_count = min_count
        self.workers = workers
        self.sample = sample
        self.negative = negative
        self.hs = hs
        self.seed = seed
        self.custom_tokenizer = custom_tokenizer
        self.model = None

    def fit(self, X, y=None):
        """
        Fit the model to the text data
        """
        self.model = Doc2Vec(
            dm=self.dm,
            vector_size=self.vector_size,
            epochs=self.epochs,
            min_count=self.min_count,
            workers=self.workers,
            sample=self.sample,
            negative=self.negative,
            hs=self.hs,
            seed=self.seed,
        )

        if self.custom_tokenizer:
            tagged_data = [
                TaggedDocument(words=self.custom_tokenizer(doc), tags=[i])
                for i, doc in enumerate(X)
            ]
        else:
            tagged_data = [
                TaggedDocument(words=doc.split(), tags=[i]) for i, doc in enumerate(X)
            ]

        self.model.build_vocab(tagged_data)
        self.model.train(
            tagged_data,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )

        return self

    def transform(self, X, y=None):
        """
        Transform the text data using the fitted model
        :param X: List of strings, a single string, or other compatible types.
        :return: List of vectors for each document
        """
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, np.ndarray):
            X = X.tolist()
        elif isinstance(X, str):
            X = [X]

        if not isinstance(X, list) or not all(isinstance(doc, str) for doc in X):
            raise ValueError("Input must be a list of strings or a single string.")

        # Transform each document to a vector
        return np.array(
            [self.model.infer_vector(doc.split()) for doc in X]
            if self.custom_tokenizer
            else [self.model.infer_vector(doc.split()) for doc in X]
        )

    def get_params(self, deep=True):
        """
        Get parameters for this estimator
        """
        return {
            "dm": self.dm,
            "vector_size": self.vector_size,
            "epochs": self.epochs,
            "min_count": self.min_count,
            "workers": self.workers,
            "sample": self.sample,
            "negative": self.negative,
            "hs": self.hs,
            "seed": self.seed,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
