import contextlib
import io
import os
import tempfile
import logging
import boto3

from s3fs import S3FileSystem
from s3fs.core import split_path

LOG = logging.getLogger('fs')


@contextlib.contextmanager
def fs_open(fname, mode="rb", **kwargs):
    """A context manager to handle file-system operations.

    See ``_open`` for more details."""
    yield _open(fname, mode=mode, **kwargs)


def _open(fname, mode="rb", **kwargs):
    """Same as native ``open`` method, but with support for S3 (via ``s3fs``).

    See ``io.open`` docstring.
    """
    if fname.startswith('s3://'):
        return S3FileSystem().open(fname, mode=mode, **kwargs)
    else:
        return io.open(fname, mode=mode, **kwargs)


def get_tempf(fname, tempdir=tempfile.gettempdir()):
    """Copy a remote resource to a temporary directory on local file system.

    Args:
        fname (str): remote filen name.
        tempdir (str): optional, temporary directory where file will be copied to.

    Returns:
        str
    """
    if fname.startswith('s3://'):
        _fname = os.path.join(tempdir, os.path.basename(fname))
        if not os.path.isfile(_fname):
            LOG.info('Copying %s ---> %s', fname, _fname)
            bucket, prefix = split_path(fname)
            with fs_open(_fname, 'wb') as f:
                boto3.client('s3').download_fileobj(bucket, prefix, f)
        return _fname
    return fname
