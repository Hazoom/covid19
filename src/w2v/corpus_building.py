import argparse
import json
import os
from typing import List

from tqdm import tqdm

from nlp import text_tokenizer
from nlp.cleaning import clean_tokenized_sentence


def _format_body(body_text: List[dict]) -> str:
    texts = [di['text'] for di in body_text]
    return " ".join(texts)


def _clean_and_tokenize_sentence(sentence) -> str:
    tokens = [str(token) for token in sentence]
    return clean_tokenized_sentence(tokens)


def _extract_sentences_from_text(text: str) -> List[str]:
    doc = text_tokenizer.tokenize_text(text)

    sentences = [_clean_and_tokenize_sentence(sentence) for sentence in doc.sents]

    return sentences


def _load_files(dirname: str) -> List[dict]:
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)

    return raw_files


def _generate_sentences(all_files: List[dict]):
    sentences = []

    print('Extracting sentences...')
    for file in tqdm(all_files):
        title_sentences = _extract_sentences_from_text(file['metadata']['title'])
        abstract_sentences = _extract_sentences_from_text(_format_body(file['abstract']))
        body_sentences = _extract_sentences_from_text(_format_body(file['body_text']))

        sentences.extend(title_sentences)
        sentences.extend(abstract_sentences)
        sentences.extend(body_sentences)
    print('Finished extracting sentences')

    return sentences


def build_corpus(dirs: List[str], output: str):
    all_sentences = []
    for dir_name in dirs:
        print(f'Loading files from directory: {dir_name} ...')
        dir_files = _load_files(dir_name)
        print(f'Finished loading files from directory: {dir_name}')
        all_sentences.extend(_generate_sentences(dir_files))

    print(f'No, of lines: {len(all_sentences)}')

    print(f'Writing TXT file to: {output}')
    with open(output, 'w+') as out_fp:
        out_fp.write("\n".join(all_sentences))


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-d', '--dirs', nargs='+', help='File directories', required=True)
    argument_parser.add_argument('-o', '--output', type=str, help='Output txt file', required=True)
    args = argument_parser.parse_args()
    build_corpus(args.dirs, args.output)
    print('Done.')


if __name__ == "__main__":
    main()
