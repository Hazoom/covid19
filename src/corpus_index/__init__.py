import re


def infer_dimension(fname):
    match = re.search(r'(\d+)d', fname)
    if not match:
        raise ValueError(f'Could not detect index dimension from {fname}.')
    dim = int(match.group(1))
    return dim


def load_corpus_index(fname, dim=None, **load_kwargs):
    """Load corpus index (either ``nmslib`` or ``annoy``) from file.

    Args:
        fname (str): filename.
        dim (int): optional, embedding dim.

    Returns:
        CorpusIndexBase
    """
    index = None
    if dim is None:
        dim = infer_dimension(fname)
    if 'nmslib' in fname:
        from corpus_index.nmslib_index import NMSLibCorpusIndex # pylint: disable=import-outside-toplevel
        index = NMSLibCorpusIndex(dim=dim)
    index.load(fname, **load_kwargs)
    return index
