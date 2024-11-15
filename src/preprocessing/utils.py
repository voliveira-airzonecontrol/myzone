import unicodedata

import numpy as np
from thefuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
import spacy


def find_best_match(
    term: str, term_list: list, return_score: bool = True
) -> tuple[str, int] or str:
    """
    Find the best match for a given code in a list of codes.
    :param term: Term to find
    :param term_list: List of terms to search
    :param return_score: Return the score of the match
    :return: Best match (and score if return_score is True)
    """

    best_match, score = process.extractOne(term, term_list, scorer=fuzz.token_set_ratio)

    return (best_match, score) if return_score else best_match


def calculate_cosine_score(vector: np.ndarray, vector_error: np.ndarray) -> float:
    return cosine_similarity(vector.reshape(1, -1), vector_error.reshape(1, -1))[0][0]


def calculate_mean_cosine_score(vector, vector_error, n=5) -> float:
    """
    Calculate the mean cosine similarity between two vectors
    :param vector: Vector
    :param vector_error: Vector to compare
    :param n: Number of iterations
    :return: Mean cosine similarity score
    """

    if vector.size == 0 or vector_error.size == 0:
        return np.nan  # Return NaN if there's no vector to compare
    cosine_scores = []
    for i in range(n):
        cosine_scores.append(calculate_cosine_score(vector, vector_error))
    return np.mean(cosine_scores)


def pre_process_text_spacy(
    docs: list[str],
    lower_case: bool = True,
    stop_words: bool = False,
    punctuation: bool = False,
    alpha: bool = False,
    lemma: bool = False,
    stemmer: bool = False,
    custom_stopwords: list[str] = None,
    md: bool = False,
) -> list[str]:
    """
    Preprocess the text by lowercasing, removing stopwords, stemming and removing punctuation
    :param docs: The text to preprocess
    :param lower_case: If True, the text will be lowercased
    :param stop_words: If True, the stopwords will be removed
    :param punctuation: If True, the punctuation will be removed
    :param lemma: If True, the text will be lemmatized
    :param stemmer: If True, the text will be stemmed
    :param custom_stopwords: A list of custom stopwords to remove
    :param md: If True, the medium model will be used
    :return: The preprocessed text
    """

    # Function to remove accents
    def remove_accents(text):
        return ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn'
        )

    nlp = spacy.load("es_core_news_md") if md else spacy.load("es_core_news_sm")

    texts = [doc for doc in nlp.pipe(docs, disable=["ner", "parser"])]
    processed_texts = []
    for doc in texts:
        tokens = []
        for token in doc:
            token_text = token.text

            if stop_words and token.is_stop:
                continue  # Skip standard stopwords
            if punctuation and token.is_punct:
                continue  # Skip punctuation
            if alpha and not token.is_alpha:
                continue  # Skip non-alphabetic characters
            if custom_stopwords:
                if token_text.lower() in custom_stopwords:
                    continue  # Skip custom stopwords
            if token.is_space:
                continue  # Skip spaces

            if lemma:
                token_text = token.lemma_  # Lemmatize the word
            if stemmer:
                raise NotImplementedError("Stemming is not implemented")
            if lower_case:
                token_text = token_text.lower()  # Lowercase the word

            # Remove accents after other transformations
            token_text = remove_accents(token_text)

            tokens.append(token_text)  # Append the processed token to the list

        processed_texts.append(" ".join(tokens))

    return processed_texts
