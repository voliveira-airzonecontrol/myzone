import numpy as np
from thefuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity


def find_best_match(
        term: str,
        term_list: list,
        return_score: bool = True
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

def calculate_cosine_score(
        vector: np.ndarray,
        vector_error: np.ndarray) -> float:

    return cosine_similarity(vector.reshape(1, -1), vector_error.reshape(1, -1))[0][0]

def calculate_mean_cosine_score(
        vector,
        vector_error,
        n=5
) -> float:

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
        cosine_scores.append(
            calculate_cosine_score(vector, vector_error)
        )
    return np.mean(cosine_scores)