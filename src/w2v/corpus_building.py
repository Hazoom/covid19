import argparse
import json
import os
from typing import List

from gensim.models.phrases import Phraser
from tqdm import tqdm

from nlp import text_tokenizer
from nlp.cleaning import clean_tokenized_sentence
from preprocessing.filtering import get_tags


def _format_body(body_text: List[dict]) -> str:
    texts = [di['text'] for di in body_text]
    return " ".join(texts)


def _clean_and_tokenize_sentence(sentence) -> str:
    tokens = [str(token) for token in sentence]
    return clean_tokenized_sentence(tokens)


def _extract_sentences_from_text(text: str, bigram_model, trigram_model) -> List[str]:
    doc = text_tokenizer.tokenize_text(text)

    cleaned_sentences = [_clean_and_tokenize_sentence(sentence) for sentence in doc.sents]

    if bigram_model and trigram_model:
        tokenzied_sentences = [sentence.split(' ') for sentence in cleaned_sentences]
        sentences_with_bigrams = bigram_model[tokenzied_sentences]
        sentences_with_trigrams = trigram_model[sentences_with_bigrams]
        results = [' '.join(sentence) for sentence in sentences_with_trigrams]
        return results

    return cleaned_sentences  # in case bi-grams and tri-grams model not provided


def _load_files(dirname: str) -> List[dict]:
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)

    return raw_files


def _generate_sentences(all_files: List[dict], bigram_model, trigram_model, filter_covid19: bool):
    sentences = []

    print('Extracting sentences...')
    for file in tqdm(all_files):
        title_text = file['metadata']['title']
        abstract_text = _format_body(file['abstract'])
        body_text = _format_body(file['body_text'])

        covid19_doc = True
        if filter_covid19:
            if not get_tags([title_text, abstract_text, body_text]):
                covid19_doc = False

        if covid19_doc:
            title_sentences = _extract_sentences_from_text(title_text, bigram_model, trigram_model)
            abstract_sentences = _extract_sentences_from_text(abstract_text, bigram_model, trigram_model)
            body_sentences = _extract_sentences_from_text(body_text, bigram_model, trigram_model)

            sentences.extend(title_sentences)
            sentences.extend(abstract_sentences)
            sentences.extend(body_sentences)
    print('Finished extracting sentences')

    return sentences


def build_corpus(dirs: List[str], output: str,  bigram_model_path: str, trigram_model_path: str, filter_covid19: bool):
    bigram_model = None
    trigram_model = None
    if bigram_model_path and trigram_model_path:
        bigram_model = Phraser.load(bigram_model_path)
        trigram_model = Phraser.load(trigram_model_path)

    all_sentences = []
    for dir_name in dirs:
        print(f'Loading files from directory: {dir_name} ...')
        dir_files = _load_files(dir_name)
        print(f'Finished loading files from directory: {dir_name}')
        all_sentences.extend(_generate_sentences(dir_files, bigram_model, trigram_model, filter_covid19))

    print(f'No, of lines: {len(all_sentences)}')

    print(f'Writing TXT file to: {output}')
    with open(output, 'w+') as out_fp:
        out_fp.write("\n".join(all_sentences))


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-d', '--dirs', nargs='+', help='File directories', required=True)
    argument_parser.add_argument('-o', '--output', type=str, help='Output txt file', required=True)
    argument_parser.add_argument('--bigram-model', type=str, help='bi-gram phrases Model', required=False)
    argument_parser.add_argument('--trigram-model', type=str, help='tri-gram phrases Model', required=False)
    argument_parser.add_argument('-f', '--filter', help='Filter out non COVID-19 articles', action="store_true")
    args = argument_parser.parse_args()
    build_corpus(args.dirs, args.output, args.bigram_model, args.trigram_model, args.filter)
    print('Done.')


if __name__ == "__main__":
    main()
