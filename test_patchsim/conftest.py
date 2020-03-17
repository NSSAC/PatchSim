"""Pytest configuration."""

import sys
import shutil
import pathlib

from pytest import fixture

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

@fixture
def workdir(tmpdir):
    """Create a working directory with PatchSim test data files."""
    basedir = pathlib.Path(__file__).resolve().parents[1] / "manual_tests"

    for fname in basedir.glob("cfg_*"):
        shutil.copy(str(basedir / fname.name), str(tmpdir))

    for fname in basedir.glob("*.txt"):
        shutil.copy(str(basedir / fname.name), str(tmpdir))

    for fname in basedir.glob("*.out.expected"):
        shutil.copy(str(basedir / fname.name), str(tmpdir))
        
    return pathlib.Path(str(tmpdir))
