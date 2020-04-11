# pylint: disable=no-member,not-callable,wrong-import-order,too-many-locals

import falcon
import murmurhash
import numpy as np
import scipy.spatial

from corpus_index.base_index import Aggregation
from db.models import Sentence
from resources.base import get_raw_sentences_from_payload, CovidResource


class CovidSimilarityResource(CovidResource):
    """Corpus Semantic Search.


        {   'results': [   {   'dist': 0.7293307781219482,
                           'id': 2021055,
                           'label': ['business_commentary_neg'],
                           'nearest': -388907830,
                           'text': 'However, if you were to exclude the Hilton '
                                   'Waikoloa Village resort, which was negatively '
                                   'impacted by Hurricane Lane and the Klauea '
                                   'volcano eruption, RevPAR performance for the '
                                   'group would have been 2.1%.'},
                       {   'dist': 0.7448862195014954,
                           'id': 125912,
                           'label': ['contract_agreement_deal_pos'],
                           'nearest': -1130842295,
                           'text': 'And then, obviously, you need to find a '
                                   'location, which is not too far away from '
                                   'downtown San Francisco and downtown L.A. And '
                                   'you need to sign up contracts.'},
                       {   'dist': 0.7739783525466919,
                           'id': 1584208,
                           'label': ['financial_results_neg'],
                           'nearest': -1130842295,
                           'text': 'This is significantly better than San '
                                   "Francisco's RevPAR decline of 3.8% in 2017 and "
                                   'again that is for the city of San Francisco '
                                   'not the metropolitan market.'},
                       {   'dist': 0.7758655548095703,
                           'id': 628653,
                           'label': ['deception_revenue_neg', 'weather_neg'],
                           'nearest': -1130842295,
                           'text': 'So we looked at our Washington, D.C. '
                                   'restaurant or our Houston restaurant, you can '
                                   'clearly see significant negative numbers '
                                   'coming from the one-off rollovers and we have '
                                   'other restaurants that are less impacted by '
                                   'weather that are trending positively '
                                   'year-to-date.'},
                       {   'dist': 0.7799857258796692,
                           'id': 1553453,
                           'label': ['business_commentary_pos'],
                           'nearest': -388907830,
                           'text': 'So you need to see it both, and we see very '
                                   "positive developments let's in the Southern "
                                   'cities like Bangalore or Hyderabad.'},
                       {   'dist': 0.7810379862785339,
                           'id': 1076013,
                           'label': ['business_commentary_pos'],
                           'nearest': -1130842295,
                           'text': 'We are well positioned with our assets and our '
                                   'Play New Jersey brand or Play NJ. And when we '
                                   'look on the state level, we now see movements '
                                   'in six states, everything from New York, '
                                   'Michigan, Illinois, Rhode Island, '
                                   'Massachusetts and Connecticut, where we all '
                                   'there is something can happen in 2018 '
                                   'already.'},
                       {   'dist': 0.7842773199081421,
                           'id': 1740568,
                           'label': ['business_commentary_neg'],
                           'nearest': -1130842295,
                           'text': 'And I think Q4 was quite slow for launches in '
                                   'Dubai, but I noticed that Emaar had a '
                                   'successful launch of Beach Vista.'},
                       {   'dist': 0.7871336936950684,
                           'id': 194754,
                           'label': ['headwinds_tailwinds_pos'],
                           'nearest': -1130842295,
                           'text': "So, you'll see markets like Washington, D.C. "
                                   'or Chicago improving.'},
                       {   'dist': 0.7891233563423157,
                           'id': 1431841,
                           'label': ['financial_commentary_pos'],
                           'nearest': -1130842295,
                           'text': 'Hotel Zoe, San Francisco and Skamania Lodge '
                                   'also generated healthy EBITDA increases in '
                                   '2017.'},
                       {   'dist': 0.7893946766853333,
                           'id': 400243,
                           'label': ['headwinds_tailwinds_neg'],
                           'nearest': -1130842295,
                           'text': 'Divisions where we felt the greatest impact '
                                   'for market conditions were Jacksonville, '
                                   'Phoenix and Tampa.'}],
        'sentences': [   {   'id': -1130842295,
                             'text': "I think it's like a 25% increase in "
                                     'profits.'},
                         {   'id': -484700953,
                             'text': 'So this has been units, I mean, really an '
                                     'increase in units in the stores.'},
                         {   'id': -388907830,
                             'text': 'Our bookings have continued in the last '
                                     'couple of quarters to increase.'}]}

        """

    def on_post(self, req, resp):
        """Handle POST request."""
        sentences = get_raw_sentences_from_payload(req)
        method = req.params.get('method', 'union')
        limit = int(req.params.get('limit', '10'))

        sentence_encoder = self.sentence_encoder
        corpus_index = self.corpus_index
        db_session = self.db_session

        try:
            resp.status = falcon.HTTP_200
            resp.media = self.similar_k(sentences, sentence_encoder, corpus_index, db_session, method=method,
                                        limit=limit)
        except Exception as e:
            self.logger.error('fatal error: %s', e)
            raise falcon.HTTP_INTERNAL_SERVER_ERROR('Internal Server Error')

    @staticmethod
    def similar_k(input_sentences, sentence_encoder, corpus_index, db_session, limit=10, method='union',
                  group_by='cosine'):
        """Find similar sentences.

        Args:
            input_sentences (str/list[str]): one or more input sentences.
            sentence_encoder  : encoder
            limit (int): limit result set size to ``limit``.
            corpus_index : type of corpus where to fetch the suggestions from
            db_session  : Database to get neighbors from
            method (str): aggregation method ('union', 'mean', 'pc1', 'pc2').
            group_by (str): distance metric to use to group the result set. Default is 'cosine'.

        Returns:
            list<dict>
        """
        res = []
        nearest = dict()

        if method == 'textrank':
            from nlp.textrank import calc_textrank  # pylint: disable=import-outside-toplevel
            _, _, _, phrase_list = calc_textrank(input_sentences, num_phrases=5, preprocess=True)
            input_sentences = [' '.join(phrase[0] for phrase in phrase_list)]
            method = Aggregation.UNION

        embeddings = sentence_encoder.encode(input_sentences)
        indices = [murmurhash.hash(sent) for sent in input_sentences]

        for idx, dist in corpus_index.knn_query_batch(embeddings, ids=indices, limit=limit, method=method):
            if idx not in nearest:
                nearest[idx] = dist
            else:
                nearest[idx] = min(nearest[idx], dist)

        for sentence in db_session.query(Sentence).filter(Sentence.id.in_(nearest.keys())).all():
            sentence_dict = sentence.to_dict()
            encoding = sentence_encoder.encode(sentence.sentence)
            distances = scipy.spatial.distance.cdist(encoding, embeddings, group_by)
            nearest_idx = int(np.argmax(distances))
            sentence_dict['nearest'] = indices[nearest_idx]
            sentence_dict['dist'] = nearest[sentence.id]
            res.append(sentence_dict)

        return {
            'results': sorted(res, key=lambda x: x['dist']),
            'sentences': [
                {
                    'id': sent_id,
                    'text': sent
                } for sent_id, sent in zip(indices, input_sentences)
            ]
        }
