# pylint: disable=invalid-name
from config import data_dir, db_connection
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from utils.fs_utils import get_tempf

Base = declarative_base()


def get_db_connection_string(conn, corpus='earnings_calls'):
    """If ``conn`` is a S3 URL, download the file to temporary directory
    and load it from here as sqlite table."""
    if conn.startswith('s3://'):
        return f"sqlite:///{get_tempf(conn, data_dir + '/' + corpus)}"
    return db_connection


def get_session(conn=None, corpus='earnings_calls'):
    """Init DB Session.

    By default, connection string is defined as ```$DB_CONNECTION```.
    """
    connect_args = {}
    conn = get_db_connection_string(conn, corpus)
    if conn.startswith('sqlite:///'):
        connect_args.update({'check_same_thread': False})
    engine = create_engine(conn, connect_args=connect_args)
    Session = sessionmaker(bind=engine)
    return Session()
