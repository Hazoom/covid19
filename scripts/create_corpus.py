import argparse
import os
import time

import argcomplete
import numpy as np
import pandas as pd

from config import data_dir
from corpus_index.nmslib_index import NMSLibCorpusIndex
from encoders import get_encoder
from utils.data_utils import svd_components


def embed_sentences(sentences, args):
    """Create sentence embeddings from raw sentences.

    Save the output to disk, either as ```nmslib.index``` or ``np.array``.

    Args:
        sentences (np.array): input sentences.
        args (argparse.Namespace): command line arguments.
    """
    encoder = get_encoder(args.encoder)
    start = time.time()
    embeddings = encoder.encode(sentences)
    encoding_time = time.time() - start
    embeddings_dim = embeddings[0].shape[0]
    print(f'-I- Done in {encoding_time} seconds.')

    if args.indices:
        embeddings_ids = np.loadtxt(args.indices, dtype=int, delimiter='\n')
    else:
        embeddings_ids = None

    # remove principal components
    if args.encoder == 'simple_encoder' and args.remove_components > 0 and encoder.components_ is None:
        components_ = svd_components(embeddings, n_components=args.remove_components)
        embeddings -= embeddings.dot(components_.transpose()).dot(components_)

        components_fname = os.path.join(
            args.output_dir, f'{args.encoder.replace("_", "-")}-{embeddings_dim}d-components.npy')
        print(f'-I- Saving PCA components to {components_fname}')
        np.save(components_fname, components_)

    # persist to disk (either .npy / nmslib / annoy )
    if args.index:
        target = os.path.join(
            args.output_dir,
            f'{args.encoder.replace("_", "-")}-{args.index}-{embeddings_dim}d.bin')

        index = NMSLibCorpusIndex(dim=embeddings_dim)

        print(f'-I- Building {args.index} ANN Index: {target}')
        start = time.time()
        index.add_dense(embeddings, ids=embeddings_ids)
        index.save(target)
        indexing_time = time.time() - start
        print(f'-I- Done in {int(indexing_time / 60)} minutes.')
    else:
        target = os.path.join(args.output_dir, f'{args.encoder.replace("_", "-")}-{embeddings_dim}d.npy')
        print(f'-I- Saving embeddings to file: {target}')
        np.save(target, embeddings)


def parse_arguments():
    parser = argparse.ArgumentParser(description='embed sentences')

    parser.add_argument('corpus_file',
                        help="corpus file for training")

    parser.add_argument('--encoder',
                        default='simple_encoder',
                        choices=['simple_encoder', 'use_encoder'],
                        help="Sentence encoder.")

    parser.add_argument('--remove-components',
                        default=0,
                        type=int,
                        help="remove principal components (relevant only to simple encoder)")
    parser.add_argument('--index',
                        default=None,
                        choices=[None, 'nmslib'],
                        help='Build A nmslib index')

    parser.add_argument('--indices',
                        default=None,
                        help="Path to a file with sentence ids.")

    parser.add_argument('--output-dir',
                        default=data_dir,
                        help="Path to a directory where model will be saved.")

    argcomplete.autocomplete(parser)
    options = parser.parse_args()

    if not os.path.isfile(options.corpus_file):
        raise IOError(f'-E- file not found: {options.corpus_file}')

    return options


if __name__ == '__main__':
    args = parse_arguments()  # pylint: disable=invalid-name
    print(f'-I- options: {args}')

    print('-I- Loading sentences from CSV files')

    sentences = pd.read_csv(args.corpus_file, index_col=0)['cleaned_sentence'].values

    print(f'-I- Encoding {sentences.shape[0]} sentences')
    embed_sentences(sentences, args)
