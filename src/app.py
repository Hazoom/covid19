import os

import falcon

from config import data_dir
from corpus_index import load_corpus_index
from db import get_session
from encoders import get_encoder
from resources.similarity import CovidSimilarityResource


def create_app(env=None):
    """Create falcon WSGI application.

    Args:
        env (dict-like): environment variables.

    Returns:
        falcon.API
    """
    # By default we use the system environment.
    env = env or os.environ

    # Init encoder
    encoder_name = env.get('ENCODER', 'simple_encoder')
    print(f"Init  Simple Encoder - {encoder_name}")
    sentence_encoder = get_encoder(encoder_name)

    # Init corpus index
    corpus_index_fname = env.get('CORPUS_INDEX',
                                 os.path.join(data_dir, 'simple-encoder-nmslib-100d.bin'))
    print(f"Init Corpus Index - {corpus_index_fname}")
    corpus_index = load_corpus_index(corpus_index_fname)

    # Init DB
    print('Init Covid DB')
    db_session = get_session(conn=os.getenv('DB_CONNECTION', os.path.join(data_dir, 'covid19.sqlite')))

    # Create WSGI application
    app = falcon.API()  # pylint: disable=invalid-name

    # Init routes
    app.add_route('/similar',
                  CovidSimilarityResource(corpus_index, sentence_encoder, db_session))

    print('*** WSGI application is ready ***')
    return app


if __name__ == '__main__':
    from wsgiref import simple_server

    httpd = simple_server.make_server('127.0.0.1', 5000, create_app())  # pylint: disable=invalid-name
    httpd.serve_forever()
