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
