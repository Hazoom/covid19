import csv
import argparse
import pandas as pd

CSV_HEADERS = ["organization_name", "ticker", "article_id", "article_source", "article_date", "article_title",
               "article_text"]


def _save_df(df_articles: pd.DataFrame, output_file: str):
    df_articles.to_csv(output_file, quoting=csv.QUOTE_ALL, index=False, line_terminator='\n')


def create_vip_csv(input_file: str, output_file: str):
    files_df = pd.read_csv(input_file)
    files_df['title'] = files_df['title'].astype(str)
    files_df['abstract'] = files_df['abstract'].astype(str)
    files_df['text'] = files_df['text'].astype(str)

    articles = []
    for _, row in files_df.iterrows():
        article = ['?',
                   '?',
                   row['paper_id'],
                   row['source'],
                   row['publish_time'],
                   row['title'],
                   row['abstract'] + row['text']]
        articles.append(article)
    df_articles = pd.DataFrame(data=articles, columns=CSV_HEADERS)

    print(f'Writing CSV file to: {output_file}')
    _save_df(df_articles, output_file)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--input-file', type=str, help='Input CSV file', required=True)
    argument_parser.add_argument('-o', '--output-file', type=str, help='Output CSV file', required=True)
    args = argument_parser.parse_args()
    create_vip_csv(args.input_file, args.output_file)
    print('Done.')


if __name__ == "__main__":
    main()
