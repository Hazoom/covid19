from abc import ABC, abstractmethod

import numpy as np

from utils.data_utils import linalg_pca


class Aggregation:
    UNION = 'union'
    AVG = 'average'
    MEAN = 'mean'
    PC_1 = 'pc_1'
    PC_2 = 'pc_2'


class CorpusIndexBase(ABC):
    _knn_batch_method = frozenset([Aggregation.UNION,
                                   Aggregation.AVG,
                                   Aggregation.MEAN,
                                   Aggregation.PC_1,
                                   Aggregation.PC_2])

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("should be implemented by subclass")

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError("should be implemented by subclass")

    @abstractmethod
    def load(self, fname, **kwargs):
        """Load an index from disk.

        Args:
            fname (str): filename.
            kwargs: additional keyword arguments.
        """
        raise NotImplementedError("should be implemented by subclass")

    @abstractmethod
    def save(self, fname, **kwargs):
        """Save index to disk.

        Args:
            fname (str): filename.
            kwargs: additional keyword arguments.
        """
        raise NotImplementedError("should be implemented by subclass")

    @abstractmethod
    def create_index(self):
        """Create ANN Index."""
        raise NotImplementedError("should be implemented by subclass")

    @abstractmethod
    def get_vector_by_id(self, idx):
        """Get vector from index by id.

        Args:
            idx (int): vector id.

        Returns:
            np.array.
        """
        raise NotImplementedError("should be implemented by subclass")

    def get_embeddings(self):  # pylint: disable=inconsistent-return-statements
        """Get embedding matrix of the index.

        Returns:
            np.array.
        """
        index_size = len(self)
        if not index_size:
            return
        return np.stack([self.get_vector_by_id(id) for id in range(index_size)])

    @abstractmethod
    def add_dense(self, dense, ids=None):
        """Add a batch of vectors to the index.

        Args:
            dense (array-like): array like of vectors (each is a ``np.array``).
            ids (array-like): array like of indices (each is a ``int``).
        """
        raise NotImplementedError("should be implemented by subclass")

    @abstractmethod
    def knn_query(self, vec, ids=None, limit=10):
        """Find a set of approximate nearest neighbors to ``vec``.

        Args:
            vec (np.array): input vector.
            ids (list): optional, list of indices to filter out from result.
            limit (int): optional, limit result set size.

        Returns:
            list[tuple] = (neighbor_id, distance)
        """
        raise NotImplementedError("should be implemented by subclass")

    @abstractmethod
    def _knn_query_batch(self, dense, ids=None, limit=10):
        """Find the union set of approximate nearest neighbors to ``dense``.

        Args:
            dense (array-like): array like of vectors (each is a ``np.array``).
            ids (list): optional, list of indices to filter out from result.
            limit (int): optional, limit result set size.

        Returns:
            list[tuple] = (neighbor_id, distance)
        """
        raise NotImplementedError("should be implemented by subclass")

    def knn_query_batch(self, dense, ids=None, limit=10, method='union'):
        """Find a set of approximate nearest neighbors to ``dense``.

        If ```method``` is 'union', than this set will be the top-slice of the union set of all nearest neighbors.

        If ```method``` is 'mean', than this set equals to
            ```self.knn_query(np.mean(dense, axis=0), ids=ids, limit=limit)```.

        If ```method``` is 'pc_1', than this set equals to
            ```self.knn_query(_linalg_pca(dense)[0], ids=ids, limit=limit)```.

        If ```method``` is 'pc_2', than this set equals to
            ```self.knn_query(_linalg_pca(dense)[1], ids=ids, limit=limit)```.

        Args:
            dense (array-like): array like of vectors (each is a ``np.array``).
            ids (iterable): optional, list of indices to filter out from result.
            limit (int): optional, limit result set size.
            method (str): optional

        Returns:
            list[tuple] = (neighbor_id, distance)
        """
        self._check_batch_method(method)

        if method in ('average', 'mean'):
            return self.knn_query(np.mean(dense, axis=0), ids=ids, limit=limit)
        elif method == 'pc_1':
            return self.knn_query(linalg_pca(dense)[0], ids=ids, limit=limit)
        elif method == 'pc_2':
            return self._knn_query_batch(linalg_pca(dense)[:1], ids=ids, limit=limit)
        elif method == 'union':
            return self._knn_query_batch(dense, ids=ids, limit=limit)
        else:  # union
            return self._knn_query_batch(dense, ids=ids, limit=limit)

    def _check_batch_method(self, method):
        assert method in self._knn_batch_method, f"Invalid KNN batch method: {method}"

    def _check_dim(self, dense):
        dim = getattr(self, 'dim', None)
        if dim:
            if len(dense.shape) == 2:
                dense_dim = dense.shape[1]
            else:
                dense_dim = dense.shape[0]

            assert dim == dense_dim, f"expected dense vectors shape to be {dim}, got {dense_dim} instead."
