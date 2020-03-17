"""Test utility functions."""

import os
import pathlib
import filecmp
from contextlib import contextmanager

def assert_equal_files(filea, fileb, dirname=None):
    """Assert that the files in the given directory are equal.

    Parameters
    ----------
    filea : str
        Path to the file file
    fileb : str
        Path to the second file
    dirname : str, optional
        If presend the filea and fileb are considered relative to dirname
    """
    filea = str(filea)
    fileb = str(fileb)

    if dirname is not None:
        dirname = pathlib.Path(str(dirname))
        filea = str(dirname / filea)
        fileb = str(dirname / fileb)

    assert filecmp.cmp(filea, fileb, shallow=False)

@contextmanager
def chdir_context(dirname):
    """Return a context manager with given current directory.

    Parameters
    ----------
    dirname : str
        Path to directory to switch to
    """
    curdir = os.getcwd()
    try:
        os.chdir(str(dirname))
        yield
    finally:
        os.chdir(curdir)
