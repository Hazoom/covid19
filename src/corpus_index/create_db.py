import argparse
import os

import pandas as pd
from sqlalchemy import create_engine

from config import data_dir


def create_db(df, connection_string, table_name='sentences', batch_size=1000):  # pylint: disable=invalid-name
    """Save corpus to a database table.

    Args:
        df (pd.DataFrame): corpus data.
        connection_string (str): database connection string (https://docs.sqlalchemy.org/en/13/core/engines.html).
        table_name (str): optional, table name.
        batch_size (int): optional, insert batch size.
    """

    engine = create_engine(connection_string)
    df.to_sql(table_name,
              engine,
              index=True,
              index_label='sentence_id',
              if_exists='replace',
              chunksize=batch_size)


def parse_arguments():
    parser = argparse.ArgumentParser(description='build corpus from a raw dataset')

    parser.add_argument('input',
                        help="earning calls dataset (csv format)")

    parser.add_argument('--sqlite',
                        action='store_true',
                        help="Save corpus to SQLite")

    parser.add_argument('--output-dir',
                        default=data_dir,
                        help="Path to a directory where model will be saved.")

    options = parser.parse_args()

    if not os.path.isfile(options.input):
        raise IOError(f'-E- file not found: {options.input}')

    return options


if __name__ == '__main__':
    args = parse_arguments()  # pylint: disable=invalid-name
    print(f'-I- options: {args}')

    print(f'-I- Loading sentencesfrom {args.input}')

    df = pd.read_csv(args.input, index_col=0)  # pylint: disable=invalid-name
    if args.sqlite:
        db_connection = os.path.join('sqlite:////',
                                     args.output_dir.lstrip(os.sep),
                                     'covid19.sqlite')

        print(f'-I- Saving to SQLite DB: {db_connection}')
        create_db(df, db_connection)  # pylint: disable=invalid-name
