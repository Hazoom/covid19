import operator
from typing import List
from datetime import datetime

import pandas as pd
from gensim.models.phrases import Phraser

from nlp.cleaning import clean_tokenized_sentence
from w2v.synonyms import Synonyms


class SearchEngine:
    def __init__(self,
                 sentences_file: str,
                 bigram_model_path: str,
                 trigram_model_path: str,
                 fasttext_model_path: str):
        print(f'Loading CSV: {sentences_file} and building mapping dictionary...')
        sentences_df = pd.read_csv(sentences_file)
        self.sentence_id_to_metadata = {}
        for row_count, row in sentences_df.iterrows():
            self.sentence_id_to_metadata[row_count] = dict(
                paper_id=row['paper_id'],
                cord_uid=row['cord_uid'],
                source=row['source'],
                publish_time=row['publish_time'],
                authors=row['authors'],
                section=row['section'],
                sentence=row['sentence'],
            )
        print(f'Finished loading CSV: {sentences_file} and building mapping dictionary')
        self.cleaned_sentences = sentences_df['cleaned_sentence'].tolist()
        print(f'Loaded {len(self.cleaned_sentences)} sentences')

        print(f'Loading bi-gram model: {bigram_model_path}')
        self.bigram_model = Phraser.load(bigram_model_path)
        print(f'Finished loading bi-gram model: {bigram_model_path}')

        print(f'Loading tri-gram model: {trigram_model_path}')
        self.trigram_model = Phraser.load(trigram_model_path)
        print(f'Finished loading tri-gram model: {trigram_model_path}')

        self.synonyms_model = Synonyms(fasttext_model_path)

    def _get_search_terms(self, keywords, synonyms_threshold):
        # clean tokens
        cleaned_terms = [clean_tokenized_sentence(keyword.split(' ')) for keyword in keywords]
        # remove empty terms
        cleaned_terms = [term for term in cleaned_terms if term]
        # create bi-grams
        terms_with_bigrams = self.bigram_model[' '.join(cleaned_terms).split(' ')]
        # create tri-grams
        terms_with_trigrams = self.trigram_model[terms_with_bigrams]
        # expand query with synonyms
        search_terms = [self.synonyms_model.get_synonyms(token) for token in terms_with_trigrams]
        # filter synonyms above threshold (and flatten the list of lists)
        search_terms = [synonym[0] for synonyms in search_terms for synonym in synonyms
                        if synonym[1] >= synonyms_threshold]
        # expand keywords with synonyms
        search_terms = list(terms_with_trigrams) + search_terms
        return search_terms

    def search(self,
               keywords: List[str],
               optional_keywords=None,
               top_n: int = 10,
               synonyms_threshold=0.7,
               keyword_weight: float = 3.0,
               optional_keyword_weight: float = 0.5) -> List[dict]:
        if optional_keywords is None:
            optional_keywords = []

        search_terms = self._get_search_terms(keywords, synonyms_threshold)

        optional_search_terms = self._get_search_terms(optional_keywords, synonyms_threshold) \
            if optional_keywords else []

        print(f'Search terms after cleaning, bigrams, trigrams and synonym expansion: {search_terms}')
        print(f'Optional search terms after cleaning, bigrams, trigrams and synonym expansion: {optional_search_terms}')

        date_today = datetime.today()

        # calculate score for each sentence. Take only sentence with at least one match from the must-have keywords
        indexes = []
        match_counts = []
        days_diffs = []
        for sentence_index, sentence in enumerate(self.cleaned_sentences):
            sentence_tokens = sentence.split(' ')
            sentence_tokens_set = set(sentence_tokens)
            match_count = sum([keyword_weight if keyword in sentence_tokens_set else 0
                               for keyword in search_terms])
            if match_count > 0:
                indexes.append(sentence_index)
                if optional_search_terms:
                    match_count += sum([optional_keyword_weight if keyword in sentence_tokens_set else 0
                                       for keyword in optional_search_terms])
                match_counts.append(match_count)
                article_date = self.sentence_id_to_metadata[sentence_index]["publish_time"]

                if article_date == "2020":
                    article_date = "2020-01-01"

                article_date = datetime.strptime(article_date, "%Y-%m-%d")
                days_diff = (date_today - article_date).days
                days_diffs.append(days_diff)

        # the bigger the better
        match_counts = [float(match_count)/sum(match_counts) for match_count in match_counts]

        # the lesser the better
        days_diffs = [(max(days_diffs) - days_diff) for days_diff in days_diffs]
        days_diffs = [float(days_diff)/sum(days_diffs) for days_diff in days_diffs]

        index_to_score = {}
        for index, match_count, days_diff in zip(indexes, match_counts, days_diffs):
            index_to_score[index] = 0.7 * match_count + 0.3 * days_diff

        # sort by score descending
        sorted_indexes = sorted(index_to_score.items(), key=operator.itemgetter(1), reverse=True)

        # take only the sentence IDs
        sorted_indexes = [item[0] for item in sorted_indexes]

        # limit results
        sorted_indexes = sorted_indexes[0: min(top_n, len(sorted_indexes))]

        # get metadata for each sentence
        results = []
        for index in sorted_indexes:
            results.append(self.sentence_id_to_metadata[index])
        return results


if __name__ == "__main__":
    from pprint import pprint

    search_engine = SearchEngine(
        "../../workspace/kaggle/covid19/data/sentences_with_metadata.csv",
        "../../workspace/kaggle/covid19/data/covid_bigram_model_v0.pkl",
        "../../workspace/kaggle/covid19/data/covid_trigram_model_v0.pkl",
        "../../workspace/kaggle/covid19/data/fasttext_no_subwords_trigrams/word-vectors-100d.txt",
    )
    terms = ["animals", "zoonotic", "farm", "spillover", "animal to human", "bats", "snakes", "exotic animals"]
    print(f"Search for terms {terms}")
    result = search_engine.search(terms, top_n=10)
    pprint(result)
