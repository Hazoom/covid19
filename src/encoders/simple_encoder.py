import os
import numpy as np
import six

from config import data_dir
from utils.fs_utils import fs_open, _open
from utils.text_utils import tokenize, clean_sentence


class WordFreq:
    """A dict-like object to hold word frequencies.

    Usage example:

    >>> freqs = WordFreq.from_counts('/path/to/word_freq.txt')
    >>> freqs['the']
    >>> 0.0505408583229405

    Once created you can use it for weighted average sentence encoding:

    >>> encoder = SentenceEncoder(..., word_freq=freqs.__getitem__)
    """

    def __init__(self, word_freq):
        self.word_freq = word_freq

    def __getitem__(self, arg):
        return self.word_freq.get(arg, 0.0)

    @classmethod
    def from_counts(cls, fname):
        total = 0
        cnts = dict()
        with fs_open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            for line in fin:
                word, cnt = line.rstrip().split(' ')
                cnt = int(cnt)
                total += cnt
                cnts[word] = cnt
        word_freq = {word: cnt / total for word, cnt in cnts.items()}
        return cls(word_freq)


class SimpleEncoder:
    def __init__(self,
                 word_embeddings: dict,
                 word_embedding_dim: int = 200,
                 preprocessor: callable = lambda s: s,
                 tokenizer: callable = lambda s: s.split(),
                 word_freq: callable = lambda w: 0.0,
                 weighted: bool = True,
                 alpha: float = 1e-3):
        """Basic sentence encoder as an average of word vectors.

        Args:
            word_embeddings (dict): map words to their vector representation.
            word_embedding_dim (int): word embedding size. default is 200.
            preprocessor (callable): optional, a callable to pre-process sentence before tokenizing into words.
            tokenizer (callable): optional, a callable which splits a sentence into words.
            word_freq (callable): optional, a callable which map a word to its frequency in range [0 - 1]
            weighted (bool): optional, whether or not to use weighted average. default is True.
            alpha (bool): smoothing alpha for Out-of-Vocab tokens.

        Usage example (1 - bag-of-words average):
        -----------------
            >>> w2v_path = '/path/to/vectors.txt'
            >>> encoder = SimpleEncoder.from_w2v(w2v_path)
            >>> encoder.encode('a sentence is here')

        Usage example (2 - Smooth Inverse Frequency average):
        -----------------
            >>> w2v_path = '/path/to/vectors.txt'
            >>> word_freq = WordFreq.from_counts('/path/to/word_freq.txt')
            >>> encoder = SimpleEncoder.from_w2v(w2v_path, weighted=True, word_freq=word_freq.__getitem__)
            >>> encoder.encode('a sentence is here')

        Usage example (3 - Smooth Inverse Frequency average + removing 1st component):
        -----------------
            >>> w2v_path = '/path/to/vectors.txt'
            >>> word_freq = WordFreq.from_counts('/path/to/word_freq.txt')
            >>> encoder = SimpleEncoder.from_w2v(w2v_path, weighted=True, word_freq=word_freq.__getitem__)
            >>> corpus = ['sentence a', 'sentence b']
            >>> emb = encoder.encode(corpus)
            >>> encoder.components_ = svd_components(emb, n_components=1)
            >>> emb = encoder.encode(corpus)  # re-calculate embeddings
            >>> encoder.encode('a sentence is here')

        """
        # word embeddings (filename)
        self.word_embeddings = word_embeddings

        # word embedding dim (e.g 200)
        self.word_embedding_dim = word_embedding_dim

        # sentence tokenizer (callable)
        self.tokenizer = tokenizer

        # preprocessor (callable)
        self.preprocessor = preprocessor

        # word frequency (callable)
        self.word_freq = word_freq

        # yes/no: tf-idf weighted average
        self.weighted = weighted

        # smoothing alpha
        self.alpha = alpha

        # principal components (pre-calc)
        self.components_ = None

    def __str__(self):
        components_dim = self.components_.shape if self.components_ is not None else None
        return (f"<SimpleEncoder(dim={self.word_embedding_dim}, "
                f"weighted={self.weighted}, "
                f"alpha={self.alpha}, "
                f"components_dim={components_dim})>")

    @classmethod
    def from_env(cls):
        """Initialize an instance of `cls` from config variables set in the environment:

            $W2V_PATH: word vectors (.txt / .vec)
            $WC_PATH: word count (.txt)
            $PC_PATH: principal components (.npy)

        Returns:
            SimpleEncoder
        """
        w2v_path = os.getenv('W2V_PATH', os.path.join(data_dir, 'word-vectors-200d.txt'))
        encoder = cls.from_w2v(w2v_path,
                               tokenizer=tokenize,
                               preprocessor=lambda sent: clean_sentence(sent,
                                                                        remove_punct=False,
                                                                        remove_numbers=False))

        word_count_path = os.getenv('WC_PATH')
        if word_count_path is not None:
            encoder.load_word_counts(word_count_path)

        principal_components_path = os.getenv('PC_PATH')
        if principal_components_path is not None:
            encoder.load_components(principal_components_path)

        return encoder

    @classmethod
    def from_w2v(cls, w2v_path, **init_kwargs):
        """Create a sentence encoder from word embeddings saved to disk.

        Args:
            w2v_path (str): filename of the word vectors.
            init_kwargs: additional keyword arguments to ```init``` method.

        Returns:
            SimpleEncoder
        """
        word_embeddings = {}
        with fs_open(w2v_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            _, dim = map(int, fin.readline().split())
            for line in fin:
                tokens = line.rstrip().split(' ')
                word_embeddings[tokens[0]] = np.array(tokens[1:], np.float32)
        return cls(word_embeddings=word_embeddings, word_embedding_dim=dim, **init_kwargs)

    def load_word_counts(self, fname):
        """Load word count file and use it for td-idf weighted average.

        Notice that ```weighted`` must be set to ```True`` in order to use it.

        Args:
            fname (str): filename.
        """
        word_freq = WordFreq.from_counts(fname)
        self.word_freq = word_freq.__getitem__

    def load_components(self, fname):
        """Load pre-computed principal components from a file.

        Args:
            fname (str): filename (e.g 'components.npy').
        """
        fd = _open(fname)  # pylint: disable=invalid-name
        self.components_ = np.load(fd)

    def encode(self, sentences) -> np.array:
        if isinstance(sentences, six.string_types):
            sentences = [sentences]
        emb = np.stack([self._encode(sentence) for sentence in sentences])
        if self.components_ is not None:
            emb = emb - emb.dot(self.components_.transpose()).dot(self.components_)
        return emb

    def _encode(self, sent: str) -> np.array:
        count = 0
        sent_vec = np.zeros(self.word_embedding_dim, dtype=np.float32)
        sent = self.preprocessor(sent)
        words = self.tokenizer(sent)
        for word in words:
            word_vec = self.word_embeddings.get(word)
            if word_vec is None:
                continue
            norm = np.linalg.norm(word_vec)
            if norm > 0:
                word_vec *= (1.0 / norm)
            if self.weighted:
                freq = self.word_freq(word)
                word_vec *= self.alpha / (self.alpha + freq)
            sent_vec += word_vec
            count += 1
        if count > 0:
            sent_vec *= (1.0 / count)
        return sent_vec
