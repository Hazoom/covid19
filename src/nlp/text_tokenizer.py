from typing import List
import spacy

from nlp import blingfire_sentence_splitter

__CACHE = {}


def tokenize_texts(texts: List[str]):
    return get_nlp_parser().pipe(texts)


def tokenize_text(text: str):
    return get_nlp_parser()(text)


def get_nlp_parser():
    if 'nlp' not in __CACHE:
        print("Loading NLP model")
        nlp = spacy.load('en_core_sci_sm', disable=['parser', 'tagger', 'ner'])
        # nlp.add_pipe(blingfire_sentence_splitter.mark_sentence_boundaries,
        #              name='mark-sentence-boundaries',
        #              first=True)
        nlp.max_length = 2000000
        __CACHE['nlp'] = nlp
    return __CACHE['nlp']


if __name__ == "__main__":
    doc = tokenize_text("Alterations in the hypocretin receptor 2 and preprohypocretin genes produce narcolepsy in "
                        "some animals.")
    print(doc.ents)
    for token in doc:
        print(token)
