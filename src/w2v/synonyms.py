import gensim.models.keyedvectors as word2vec


class Synonyms:
    def __init__(self, fasttext_model_path: str):
        print(f'Loading fasttext model: {fasttext_model_path}')
        self.model = word2vec.KeyedVectors.load_word2vec_format(fasttext_model_path)
        print(f'Finished loading fasttext model: {fasttext_model_path}')

    def get_synonyms(self, cleaned_token: str):
        return self.model.most_similar(cleaned_token)


if __name__ == "__main__":
    from pprint import pprint
    synonyms_model = Synonyms("../../workspace/kaggle/covid19/data/fasttext_no_subwords_trigrams/word-vectors-100d.txt")

    search_term = 'new_coronavirus'
    print(f'\nSynonyms of "{search_term}":')
    pprint(synonyms_model.get_synonyms(search_term))

    search_term = 'coronavirus'
    print(f'\nSynonyms of "{search_term}":')
    pprint(synonyms_model.get_synonyms(search_term))

    search_term = 'fake_news'
    print(f'\nSynonyms of "{search_term}":')
    pprint(synonyms_model.get_synonyms(search_term))
