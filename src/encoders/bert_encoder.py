# pylint: disable=global-statement,invalid-name

import argparse
import json
import os
from threading import Lock

import argcomplete
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from nlp.cache_embeddings import FiftyCache
from utils.data_utils import read_corpus

_LOCK = Lock()
_FIFTY_LOCK = Lock()

_CACHED_KEY = None
_FIFTY_CACHE = None


class BertEncoder:

    def __init__(self, model_name=os.getenv('PRETRAINED_SENTENCE_ENCODER', 'bert-base-nli-stsb-mean-tokens'),
                 encoder_config=os.getenv('BERTENCODER_DIR')):
        self.model = SentenceTransformer(model_name)
        self.encoder_params = json.load(
            open(os.path.join(encoder_config, 'bertencoder_config.json'), 'r'))

    def encode(self, sentences) -> np.array:
        if _use_cache():
            cache = get_fifty_cache()
            results = []
            for sent in sentences:
                cached_result = cache.get_from_cache(sent)
                if cached_result is None:
                    embed = self.model.encode([sent])[0]
                    results.append(embed)
                    cache.save_in_cache(sent, embed)
                else:
                    print('Sentence already cached, fetching the embedding from the cache...')
                    results.append(cached_result)
            cache.save_cache_in_disk(os.getenv('CACHE_PATH'))
            return results
        return np.stack(self.model.encode(sentences, self.encoder_params['batch_size'],
                                          self.encoder_params['show_progress_bar'])).astype('float32')


def get_fifty_cache() -> FiftyCache:
    global _FIFTY_CACHE
    if _FIFTY_CACHE is not None:
        return _FIFTY_CACHE
    with _FIFTY_LOCK:
        if _FIFTY_CACHE is None:
            if _use_cache():
                _FIFTY_CACHE = FiftyCache()
                cache_file = os.getenv('CACHE_PATH')
                if os.path.exists(cache_file):
                    _FIFTY_CACHE.load(cache_file)
        return _FIFTY_CACHE


def _use_cache():
    use_fifty_cache = bool(os.getenv('USE_FIFTY_CACHE', 'true'))
    return use_fifty_cache is not None and use_fifty_cache


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data_src", type=str,
                                 help='Input CSV file containing input data', required=False, default=None)
    argument_parser.add_argument("--corpus_file", type=str,
                                 help='Corpus file containing sentences', required=False, default=None)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    sentences = []
    encoder = BertEncoder()
    if args.data_src:
        sentences = pd.read_csv(args.data_src).sentence.values
    elif args.corpus_file:
        sentences = read_corpus(args.corpus_file).values
    embs = encoder.encode(sentences)
    print(embs)


if __name__ == "__main__":
    main()
