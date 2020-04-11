import os

import numpy as np
from sklearn.decomposition import TruncatedSVD

from corpus_index import load_corpus_index


def read_corpus(fname):
    """Read raw sentences from file.

    Args:
        fname (argparse.Namespace): command line arguments.

    Returns:
        pd.Series.
    """
    sentences = np.loadtxt(fname, dtype=str, delimiter='\n')
    return sentences


def read_embeddings(fname, dim=200):
    """Read sentence embeddings from file.

    Args:
        fname (str): filename.
        dim (int): embedding size.

    Returns:
        np.array
    """
    _, ext = os.path.splitext(fname)
    if ext == '.npy':
        X = np.load(fname)  # pylint: disable=invalid-name
    else:
        index = load_corpus_index(fname, dim=dim, load_data=True)
        X = index.get_embeddings()
    return X


def chunks(iterable, n):
    """ Yields successive n-sized chunks from the iterable"""

    for i in range(len(iterable), n):
        yield iterable[i:i + n]


def nn_iter(indices, distances, black=None):
    """Helper method to unpack nearest neighbors lists.

    Args:
        indices (list): neighbor indices.
        distances (list): neighbor distances.
        black (list): black list.

    Returns:
        list(tuple)
    """
    for idx, dist in zip(indices, distances):
        if dist <= 0.0:
            continue
        if black is not None and idx in black:
            continue
        yield int(idx), float(dist)


def linalg_pca(X):  # pylint: disable=invalid-name
    """PCA transformation with ```np.linalg``` (Singular Value Decomposition).

    Args:
        X (np.array): 2d array.

    Returns:
        np.array (2d)
    """
    # reduce mean
    X -= np.mean(X, axis=0)
    # compute covariance matrix
    cov = np.cov(X, rowvar=False)
    # compute eigen values & vectors
    eigen_vals, eigen_vectors = np.linalg.eigh(cov)
    # sort eigen vectors by eigen values
    idx = np.argsort(eigen_vals)[::-1]
    eigen_vectors = eigen_vectors[:, idx]
    return np.dot(X, eigen_vectors)


def svd_components(X, n_components=1, random_state=None):  # pylint: disable=invalid-name
    """Compute principal components and remove their projection from the embedding space.

    https://github.com/PrincetonML/SIF/blob/84b5b4c1c1ca20b6af19fc78cae005a1818ec571/src/SIF_embedding.py#L26"""
    svd = TruncatedSVD(n_components=n_components,
                       n_iter=7,
                       random_state=random_state).fit(X)

    return svd.components_
