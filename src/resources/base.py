# pylint: disable=no-member,not-callable
import json
import logging

import falcon


def get_raw_sentences_from_payload(req):
    """An helper function to extract a list of sentence from request payload.

    If ``Content-Type`` is ``application/json`` -> parse JSON payload and grab `sentences` property.

    If ``Content-Type`` is ``text/plain`` -> split payload by newline char.

    Args:
        req (falcon.request.Request): request.

    Raises:
        ``falcon.HTTPBadRequest`` - Mal-formatted request payload.
        ``falcon.HTTPError`` - internal server errors (e.g JSON parsing).

    Returns:
        list
    """
    if req.content_length in (None, 0):
        raise falcon.HTTPBadRequest('Request payload is empty.')

    content_type = req.content_type.lower()
    # curl -X POST -H "Content-Type: text/plain" --data-binary "@sentences.txt" <URL>
    if content_type == 'text/plain':
        sentences = req.bounded_stream.read().decode('utf-8').split('\n')

    # curl -X POST -H "Content-Type: application/json" --data-binary "@sentences.json" <URL>
    elif content_type == 'application/json':
        try:
            sentences = json.loads(req.bounded_stream.read().decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            raise falcon.HTTPError(falcon.HTTP_753,
                                   'Malformed JSON',
                                   'Could not decode the request body. The '
                                   'JSON was incorrect or not encoded as '
                                   'UTF-8.')
        if isinstance(sentences, dict):
            sentences = sentences['sentences']

    else:
        raise falcon.HTTP_BAD_REQUEST(f'Invalid content-type: {req.content_type}')

    sentences = [sent for sent in sentences if sent != '']
    if not sentences:
        raise falcon.HTTPBadRequest('Request payload is empty.')

    return sentences


class CovidResource:
    """Base class for all service resources."""

    def __init__(self, corpus_index, sentence_encoder, db_session, logger=None):
        self.corpus_index = corpus_index
        self.sentence_encoder = sentence_encoder
        self.db_session = db_session
        self.logger = logger or logging.getLogger(self.__class__.__name__)
