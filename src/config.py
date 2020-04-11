# pylint: disable=invalid-name
import os

# data directory
data_dir = os.getenv('DATA_DIR', os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir, 'data'
    )
))

db_connection = os.getenv('DB_CONNECTION',
                          f"sqlite:///{os.path.join(data_dir, 'covid19.sqlite')}")
