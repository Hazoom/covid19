import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class BertEncoder:
    def __init__(self, model_name='bert-base-nli-stsb-mean-tokens'):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        return np.stack(self.model.encode(sentences, show_progress_bar=True))


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data_src", type=str,
                                 help='Input CSV file containing input data', required=False, default=None)
    args = argument_parser.parse_args()
    encoder = BertEncoder()
    sentences = pd.read_csv(args.corpus_file, index_col=0)['sentence'].values
    embs = encoder.encode(sentences)
    print(embs)


if __name__ == "__main__":
    main()
