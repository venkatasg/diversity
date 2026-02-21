'''
Computes remote clique, chamfer distance, and MAUVE on embeddings for a set
of documents to understand their semantic (in embedding space) diversity.

remote_clique / chamfer_dist follow:
    Samuel Rhys Cox et al. 2021. "Directed Diversity: Leveraging Language
    Embedding Distances for Collective Creativity in Crowd Ideation".
    CHI '21. https://doi.org/10.1145/3411764.3445782

mauve_score follows:
    Pillutla et al. 2021. "MAUVE: Measuring the Gap Between Neural Text
    and Human Text using Divergence Frontiers".
    NeurIPS 2021. https://arxiv.org/abs/2102.01454
'''

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from tqdm import tqdm
from typing import List, Optional


def remote_clique(
        data: List[str],
        model: Optional[str] = 'Qwen/Qwen3-Embedding-0.6B',
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
) -> float:
    """
    Calculates the remote clique score for a set of documents (corpus-level).
    This is the average mean pairwise distance of a data instance to other instances.
    Args:
        data (List[str]): Strings to score.
        model(str, optional): Model to use for embedding. Defaults to 'Qwen/Qwen3-Embedding-0.6B'.
        verbose(bool, optional): Whether to display progress bar. Defaults to True.
        batch_size(int, optional): Batch size for embedding. Defaults to 64.
    Returns:
        float: Remote clique score.
    """
    model = SentenceTransformer(model)
    embeddings = model.encode(data, batch_size=batch_size, show_progress_bar=verbose)
    distances = cosine_distances(embeddings)
    mean_distances = np.mean(distances, axis=1)
    return np.mean(mean_distances).round(3)


def chamfer_dist(
        data: List[str],
        model: Optional[str] = 'Qwen/Qwen3-Embedding-0.6B',
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64
) -> float:
    """
    Calculates the chamfer distance for a set of documents (corpus-level).
    This is the average minimum pairwise distance of a data instance to other instances.
    Args:
        data (List[str]): Strings to score.
        model(str, optional): Model to use for embedding. Defaults to 'Qwen/Qwen3-Embedding-0.6B'.
        verbose(bool, optional): Whether to display progress bar. Defaults to True.
        batch_size(int, optional): Batch size for embedding. Defaults to 64.
    Returns:
        float: Chamfer distance.
    """
    model = SentenceTransformer(model)
    embeddings = model.encode(data, batch_size=batch_size, show_progress_bar=verbose)
    distances = cosine_distances(embeddings)
    min_distances = np.min(distances + np.eye(len(distances)) * 1e9, axis=1)
    return np.mean(min_distances).round(3)


def mauve_score(
        p_text: List[str],
        q_text: List[str],
        model: Optional[str] = 'Qwen/Qwen3-Embedding-0.6B',
        verbose: Optional[bool] = True,
        batch_size: Optional[int] = 64,
) -> float:
    """
    Calculates the MAUVE score between a reference distribution and a generated
    distribution of texts (corpus-level).

    MAUVE measures how close the generated text distribution is to the reference
    text distribution. A score of 1 indicates identical distributions; lower
    scores indicate greater divergence.

    Features are computed with a sentence-transformer model and passed directly
    to mauve.compute_mauve(), so no separate GPT-2 download is required.

    Args:
        p_text (List[str]): Reference texts (e.g. human-written).
        q_text (List[str]): Generated texts to evaluate.
        model (str, optional): Sentence-transformer model used to embed both
            distributions. Defaults to 'Qwen/Qwen3-Embedding-0.6B'.
        verbose (bool, optional): Whether to display progress bars. Defaults to True.
        batch_size (int, optional): Batch size for embedding. Defaults to 64.

    Returns:
        float: MAUVE score in [0, 1]. Higher means the two distributions are
            closer (less divergence).
    """
    import mauve as mauve_lib

    encoder = SentenceTransformer(model)
    p_features = encoder.encode(p_text, batch_size=batch_size, show_progress_bar=verbose)
    q_features = encoder.encode(q_text, batch_size=batch_size, show_progress_bar=verbose)

    out = mauve_lib.compute_mauve(
        p_features=p_features,
        q_features=q_features,
        verbose=verbose,
    )
    return round(float(out.mauve), 3)
