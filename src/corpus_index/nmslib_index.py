import operator
import nmslib
import numpy as np

from config import data_dir
from corpus_index.base_index import CorpusIndexBase
from utils.data_utils import nn_iter
from utils.fs_utils import get_tempf


class NMSLibCorpusIndex(CorpusIndexBase):
    def __init__(self, dim, metric='cosinesimil', **index_params):
        """Init ```nmslib.FloatIndex```.

        References
        -----------

        1) Installation

        https://github.com/nmslib/nmslib/tree/master/python_bindings#installation

        2) Supported metrics

        https://github.com/nmslib/nmslib/blob/master/manual/spaces.md

        3) Index params

        https://github.com/nmslib/nmslib/blob/master/manual/methods.md#graph-based-search-methods-sw-graph-and-hnsw
        """
        self.dim = dim
        self.index = nmslib.init(method='hnsw', space=metric)  # pylint: disable=c-extension-no-member
        self.index_params = index_params or {'post': 0}  # {'post': 2, 'efConstruction': 200, 'M': 25}

    def __len__(self):
        return len(self.index)

    def __repr__(self):
        return f"<NMSLibCorpusIndex(size={self.__len__()})>"

    def load(self, fname, **kwargs):
        """Load an index from disk.

        Args:
            fname (str): filename.
            kwargs: additional keyword arguments.
        """
        fname_ = get_tempf(fname, tempdir=data_dir)
        self.index.loadIndex(fname_, **kwargs)

    def save(self, fname, **kwargs):
        """Save index to disk.

        Args:
            fname (str): filename.
            kwargs: additional keyword arguments.
        """
        self.create_index()
        self.index.saveIndex(fname, save_data=True)

    def create_index(self):
        """Create ANN Index."""
        self.index.createIndex(self.index_params, print_progress=True)

    def get_vector_by_id(self, idx):
        """Get vector from index by id.

        Args:
            idx (int): vector id.

        Returns:
            np.array.
        """
        return np.array(self.index[idx], np.float32)

    def add_dense(self, dense, ids=None):
        """Add a batch of vectors to the index.

        Args:
            dense (array-like): array like of vectors (each is a ``np.array``).
            ids (array-like): array like of indices (each is a ``int``).
        """
        self._check_dim(dense)

        index_len = self.__len__()
        self.index.addDataPointBatch(
            data=dense,
            ids=ids if ids is not None else np.arange(index_len, index_len + dense.shape[0]))

    def knn_query(self, vec, ids=None, limit=10):
        """Find a set of approximate nearest neighbors to ``vec``.

        Args:
            vec (np.array): input vector.
            ids (list): optional, list of indices to filter out from result.
            limit (int): optional, limit result set size.

        Returns:
            list[tuple] = (neighbor_id, distance)
        """
        self._check_dim(vec)

        indices, distances = self.index.knnQuery(vec, k=limit * 2)
        return sorted(nn_iter(indices, distances, black=ids), key=operator.itemgetter(1))[:limit]

    def _knn_query_batch(self, dense, ids=None, limit=10):
        """Find the union set of approximate nearest neighbors to ``dense``.

        Args:
            dense (array-like): array like of vectors (each is a ``np.array``).
            ids (list): optional, list of indices to filter out from result.
            limit (int): optional, limit result set size.

        Returns:
            list[tuple] = (neighbor_id, distance)
        """
        self._check_dim(dense)

        nearest_neighbors = []
        for indices, distances in self.index.knnQueryBatch(dense, k=limit):
            for idx, dist in nn_iter(indices, distances, black=ids):
                nearest_neighbors.append((idx, dist))
        return sorted(nearest_neighbors, key=operator.itemgetter(1))[:limit]