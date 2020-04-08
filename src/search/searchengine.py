import operator
from typing import List
import pandas as pd

from nlp.cleaning import clean_tokenized_sentence


class SearchEngine:
    def __init__(self, sentences_file: str):
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

    def search(self, keywords: List[str], top_n: int = 10) -> List[dict]:
        cleaned_terms = [clean_tokenized_sentence(keyword.split(' ')) for keyword in keywords]
        cleaned_terms = [term for term in cleaned_terms if term]  # remove empty terms
        index_to_match = {}
        for sentence_index, sentence in enumerate(self.cleaned_sentences):
            match_count = sum([1 if keyword in sentence else 0 for keyword in cleaned_terms])
            if match_count > 0:
                index_to_match[sentence_index] = match_count
        sorted_indexes = sorted(index_to_match.items(), key=operator.itemgetter(1), reverse=True)
        sorted_indexes = [item[0] for item in sorted_indexes]
        sorted_indexes = sorted_indexes[0: min(top_n, len(sorted_indexes))]
        results = []
        for index in sorted_indexes:
            results.append(self.sentence_id_to_metadata[index])
        return results


if __name__ == "__main__":
    from pprint import pprint

    search_engine = SearchEngine("../../workspace/kaggle/covid19/data/sentences_with_metadata.csv")
    terms = ["animals", "zoonotic", "farm", "spillover", "animal to human", "bats", "snakes", "exotic animals"]
    print(f"Search for terms {terms}")
    result = search_engine.search(terms, top_n=10)
    pprint(result)
