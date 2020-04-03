import json
import os
from typing import List
import argparse
from copy import deepcopy

import pandas as pd
from tqdm import tqdm


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


def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}

    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"

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


def get_tags(sections):
    """
    Credit: https://www.kaggle.com/davidmezzetti/cord-19-analysis-with-sentence-embeddings

    Searches input sections for matching keywords. If found, returns the keyword tag.
    Args:
        sections: list of text sections
    Returns:
        tags
    """

    keywords = ["2019-ncov", "2019 novel coronavirus", "coronavirus 2019", "coronavirus disease 19", "covid-19",
                "covid 19", "ncov-2019",
                "sars-cov-2", "wuhan coronavirus", "wuhan pneumonia", "wuhan virus"]

    tags = None
    for text in sections:
        if any(x in text.lower() for x in keywords):
            tags = "COVID-19"

    return tags


def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)

    return raw_files


def generate_clean_df(all_files):
    cleaned_files = []

    for file in tqdm(all_files):
        title = file['metadata']['title']
        abstract = format_body(file['abstract'])
        body_text = format_body(file['body_text'])
        features = [
            file['paper_id'],
            title,
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'],
                           with_affiliation=True),
            abstract,
            body_text,
            get_tags([title, abstract, body_text])
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 'tag']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    print(f'No. of files before filtering COVID-19 files: {len(clean_df)}')
    clean_df = clean_df[clean_df['tag'] == 'COVID-19']
    print(f'No. of files after filtering COVID-19 files: {len(clean_df)}')

    return clean_df


def build_csv(dirs: List[str], output: str):
    all_df = None
    for dir_name in dirs:
        print(f'Loading files from directory: {dir_name} ...')
        dir_files = load_files(dir_name)
        print(f'Finished loading files from directory: {dir_name}')
        if all_df is None:
            all_df = generate_clean_df(dir_files)
        else:
            all_df.append(generate_clean_df(dir_files))

    print(f'All files DataFrame shape: {all_df.shape}')

    print(f'Writing CSV file to: {output}')
    with open(output, 'w+') as out_fp:
        all_df.to_csv(out_fp, index=False)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-d', '--dirs', nargs='+', help='File directories', required=True)
    argument_parser.add_argument('-o', '--output', type=str, help='Output CSV file', required=True)
    args = argument_parser.parse_args()
    build_csv(args.dirs, args.output)
    print('Done.')


if __name__ == "__main__":
    main()
