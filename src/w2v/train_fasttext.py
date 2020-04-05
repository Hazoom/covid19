import argparse
import errno
import os

try:
    import fasttext
except ImportError:
    raise ImportError("""``fasttext`` is not installed. please follow the installation guide -> 
                      https://github.com/facebookresearch/fastText#building-fasttext-for-python.""")


def bin_to_vec(model, fname):
    """Save w2v model weights to file.

    Output file format:
    line 1: <N D>, where N = number of lines, D = word vector dimension.
    line 2-N+1: <WORD x1 x2 .... xd>


    Args:
        model (`fasttext._FastText`): w2v model.
        fname (str): output file.

    See ```https://github.com/facebookresearch/fastText/blob/master/python/doc/examples/bin_to_vec.py```.
    """
    with open(fname, 'w') as f:
        words = model.get_words()
        f.write(f'{len(words)} {model.get_dimension()}\n')
        for w in words:
            v = model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                f.write(f'{w}{vstr}\n')
            except IOError as e:
                if e.errno == errno.EPIPE:
                    pass


def bin_to_word_count(model, fname):
    """Save w2v model word counts to file.

    Args:
        model (`fasttext._FastText`): w2v model.
        fname (str): output file.
    """
    with open(fname, 'w') as f:
        for word, word_freq in zip(*model.get_words(include_freq=True)):
            f.write(f'{word} {word_freq}\n')


def train_unsupervised(args):

    # https://fasttext.cc/docs/en/unsupervised-tutorial.html
    model = fasttext.train_unsupervised(args.input,
                                        lr=args.lr,
                                        minCount=args.min_count,
                                        epoch=args.epoch,
                                        minn=args.minn,
                                        maxn=args.maxn,
                                        dim=args.dim,
                                        ws=args.ws)

    if not os.path.isdir(args.output_dir):
        print(f'Creating output directory: {args.output_dir}')
        os.makedirs(args.output_dir)

    model_fname = os.path.join(args.output_dir, 'model.bin')
    print(f'Saving model to: {model_fname}')
    model.save_model(model_fname)

    vec_fname = os.path.join(args.output_dir, f'word-vectors-{args.dim}d.txt')
    print(f'Saving word vectors to: {vec_fname}')
    bin_to_vec(model, vec_fname)

    count_fname = os.path.join(args.output_dir, f'word-counts.txt')
    print(f'Saving word count to: {count_fname}')
    bin_to_word_count(model, count_fname)


def main():
    argument_parser = argparse.ArgumentParser(description='train w2v with fassttext unsupervised model')

    argument_parser.add_argument('--input',
                                 help="corpus file for training (each line is a sentence)")

    argument_parser.add_argument('--output-dir',
                                 help="Path to a directory where model will be saved.")

    argument_parser.add_argument('--epoch',
                                 default=5,
                                 type=int,
                                 help="num of training epochs.")

    argument_parser.add_argument('--min-count',
                                 default=5,
                                 type=int,
                                 help="minimal number of word occurrences")

    argument_parser.add_argument('--minn',
                                 default=0,
                                 type=int,
                                 help="min length of char ngram")

    argument_parser.add_argument('--maxn',
                                 default=0,
                                 type=int,
                                 help="max length of char ngram")

    argument_parser.add_argument('--ws',
                                 default=5,
                                 type=int,
                                 help="size of the context window")

    argument_parser.add_argument('--dim',
                                 default=100,
                                 type=int,
                                 help="size of word vectors")

    argument_parser.add_argument('--lr',
                                 default=0.05,
                                 type=float,
                                 help="learning rate")

    args = argument_parser.parse_args()
    train_unsupervised(args)
    print('Done.')


if __name__ == "__main__":
    main()
