# pylint: disable=no-member
import logging

import falcon

from resources.base import CovidResource

LOG = logging.getLogger('gunicorn.error')


class CovidStatusResource(CovidResource):

    def on_get(self, resp):
        """Handle POST request."""

        sentence_encoder = self.sentence_encoder
        db_session = self.db_session

        try:
            self.check_status(sentence_encoder, db_session)
            resp.status = falcon.HTTP_200
            resp.media = {
                'status': 'ok',
                'sentence_encoder': str(sentence_encoder),
                'db': str(db_session.bind.url)
            }
        except Exception as e:  # pylint: disable=broad-except
            resp.status = falcon.HTTP_INTERNAL_SERVER_ERROR
            resp.media = {
                'error': {
                    'name': e.__class__.__name__,
                    'reason': str(e)
                }
            }

    @staticmethod
    def check_status(sentence_encoder, db_session):
        db_session.execute('SELECT 1').first()  # db sanity check
        sentence_encoder.encode('this is a test')  # sentence encoder sanity check
