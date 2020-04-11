import json
import os
from typing import List
import argparse
from copy import deepcopy
from gensim.models.phrases import Phraser

import pandas as pd
from tqdm import tqdm

from preprocessing.filtering import get_tags
from nlp import text_parsing
from nlp.cleaning import clean_tokenized_sentence


# Helper Functions by https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv


def format_name(author):
    middle_name = " ".join(author['middle'])

    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))

    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)


def format_authors(authors, with_affiliation=False):
    name_ls = []

    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)

    return ", ".join(name_ls)


def _get_text_from_sections(body_text):
    texts = [di['text'] for di in body_text]
    return " ".join(texts)


def _clean_and_tokenize_sentence(sentence) -> str:
    tokens = [str(token) for token in sentence]
    return clean_tokenized_sentence(tokens)


def _clean_sentences(sentences) -> List[str]:
    return [_clean_and_tokenize_sentence(sentence) for sentence in sentences]


def _extract_sentences_from_text(text: str) -> List[str]:
    doc = text_parsing.parse_text(text)

    return list(doc.sents)


def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}

    for section, text in texts:
        texts_di[section] += (text + "\r\n")

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\r\n\r\n"
        body += text
        body += "\r\n\r\n"

    return body


def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []

    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'],
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)


def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)

    return raw_files


def _add_line_for_sentence_level(section_type,
                                 text,
                                 sentences_lines,
                                 file,
                                 sha_to_properties,
                                 title_text,
                                 bigram_model,
                                 trigram_model) -> None:
    sentences = _extract_sentences_from_text(text)
    cleaned_sentences = _clean_sentences(sentences)
    tokenzied_sentences = [sentence.split(' ') for sentence in cleaned_sentences]
    sentences_with_bigrams = bigram_model[tokenzied_sentences]
    sentences_with_trigrams = trigram_model[sentences_with_bigrams]
    cleaned_sentences = [' '.join(sentence) for sentence in sentences_with_trigrams]
    features = [
        file['paper_id'],
        sha_to_properties[file['paper_id']]['cord_uid'],
        sha_to_properties[file['paper_id']]['source'],
        sha_to_properties[file['paper_id']]['publish_time'],
        title_text,
        format_authors(file['metadata']['authors']),
        section_type,
        [str(sentence) for sentence in sentences],
        cleaned_sentences
    ]
    sentences_lines.append(features)


def generate_df_sentence_level(all_files: List[dict],
                               sha_to_properties: dict,
                               filter_covid19: bool,
                               bigram_model,
                               trigram_model) -> pd.DataFrame:
    sentences_lines = []

    for file in tqdm(all_files):
        title_text = file['metadata']['title']
        abstract_text = _get_text_from_sections(file['abstract'])
        body_text = _get_text_from_sections(file['body_text'])

        covid19_doc = True
        if filter_covid19:
            if not get_tags([title_text, abstract_text, body_text]):
                covid19_doc = False

        if covid19_doc:
            _add_line_for_sentence_level(
                'abstract', abstract_text, sentences_lines, file, sha_to_properties, title_text, bigram_model,
                trigram_model
            )
            _add_line_for_sentence_level(
                'body', body_text, sentences_lines, file, sha_to_properties, title_text, bigram_model,
                trigram_model
            )

    print(f'No. of lines before breaking to sentences: {len(sentences_lines)}')

    col_names = ['paper_id', 'cord_uid', 'source', 'publish_time', 'title', 'authors',
                 'section', 'sentence', 'cleaned_sentence']

    sentences_to_df = []
    for sentences_line in sentences_lines:
        sentences = sentences_line[-2]
        cleaned_sentences = sentences_line[-1]
        for sentence, cleaned_sentence in zip(sentences, cleaned_sentences):
            if sentence and cleaned_sentence:
                sentences_to_df.append(sentences_line[0: len(col_names) - 2] + [sentence, cleaned_sentence])

    sentences_df = pd.DataFrame(sentences_to_df, columns=col_names)

    print(f'No. of lines after breaking to sentences: {len(sentences_df)}')

    return sentences_df


def generate_df(all_files: List[dict], sha_to_properties: dict, filter_covid19: bool) -> pd.DataFrame:
    cleaned_files = []

    for file in tqdm(all_files):
        title_text = file['metadata']['title']
        abstract_text = format_body(file['abstract'])
        body_text = format_body(file['body_text'])

        covid19_doc = True
        if filter_covid19:
            if not get_tags([title_text, abstract_text, body_text]):
                covid19_doc = False

        if covid19_doc:
            features = [
                file['paper_id'],
                sha_to_properties[file['paper_id']]['cord_uid'],
                sha_to_properties[file['paper_id']]['source'],
                sha_to_properties[file['paper_id']]['publish_time'],
                title_text,
                format_authors(file['metadata']['authors']),
                format_authors(file['metadata']['authors'],
                               with_affiliation=True),
                abstract_text,
                body_text
            ]
            cleaned_files.append(features)

    col_names = ['paper_id', 'cord_uid', 'source', 'publish_time', 'title', 'authors',
                 'affiliations', 'abstract', 'text']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    print(f'No. of files: {len(clean_df)}')

    return clean_df


def _build_metadata_dict(metadata):
    metadata_df = pd.read_csv(metadata)
    metadata_df['sha'] = metadata_df['sha'].astype(str)
    sha_to_properties = {}
    for _, row in metadata_df.iterrows():
        for sha in row['sha'].split('; '):
            if sha not in sha_to_properties:
                sha_to_properties[sha] = {'cord_uid': row['cord_uid'],
                                          'source': row['source_x'],
                                          'publish_time': row['publish_time']}
    return sha_to_properties


def build_csv(metadata: str,
              dirs: List[str],
              output: str,
              bigram_model_path: str,
              trigram_model_path: str,
              filter_covid_19: bool,
              sentences: bool):
    print(f'Building metadata dictionary from file: {metadata} ...')
    sha_to_properties = _build_metadata_dict(metadata)
    print(f'Finished building metadata dictionary from file: {metadata}')

    print(f'Filtering only COVID-19 articles: {filter_covid_19}')

    bigram_model = Phraser.load(bigram_model_path)
    trigram_model = Phraser.load(trigram_model_path)

    all_df = None
    for dir_name in dirs:
        print(f'Loading files from directory: {dir_name} ...')
        dir_files = load_files(dir_name)
        print(f'Finished loading files from directory: {dir_name}')
        if sentences:
            clean_df = generate_df_sentence_level(
                dir_files, sha_to_properties, filter_covid_19, bigram_model, trigram_model
            )
        else:
            clean_df = generate_df(dir_files, sha_to_properties, filter_covid_19)

        if all_df is None:  # first call
            all_df = clean_df
        else:
            all_df = all_df.append(clean_df)

    all_df.fillna("", inplace=True)

    print(f'All files DataFrame shape: {all_df.shape}')

    print(f'Writing CSV file to: {output}')
    with open(output, 'w+') as out_fp:
        all_df.to_csv(out_fp, index=True)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-m', '--metadata', type=str, help='Metadata CSV file', required=True)
    argument_parser.add_argument('-d', '--dirs', nargs='+', help='File directories', required=True)
    argument_parser.add_argument('-o', '--output', type=str, help='Output CSV file', required=True)
    argument_parser.add_argument('--bigram-model', type=str, help='bi-gram phrases Model', required=True)
    argument_parser.add_argument('--trigram-model', type=str, help='tri-gram phrases Model', required=True)
    argument_parser.add_argument('--no-filter-covid-19', help='No filter COVID-19 docs', action="store_true")
    argument_parser.add_argument('--sentences', help='Each line is a sentence in text', action="store_true")
    args = argument_parser.parse_args()
    build_csv(
        args.metadata,
        args.dirs,
        args.output,
        args.bigram_model,
        args.trigram_model,
        not args.no_filter_covid_19,
        args.sentences
    )
    print('Done.')


if __name__ == "__main__":
    main()
