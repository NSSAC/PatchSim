"""Pytest configuration."""

import sys
import shutil
import pathlib
import subprocess

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parent))


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
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


@pytest.fixture
def workdir_us_county(tmpdir):
    """Create a working directory with PatchSim test data files."""
    basedir = pathlib.Path(__file__).resolve().parents[1] / "manual_tests"

    for fname in basedir.glob("cfg_*"):
        shutil.copy(str(basedir / fname.name), str(tmpdir))

    for fname in basedir.glob("*.txt"):
        shutil.copy(str(basedir / fname.name), str(tmpdir))

    for fname in basedir.glob("*.out.expected"):
        shutil.copy(str(basedir / fname.name), str(tmpdir))

    shutil.copy(str(basedir / "US_county.zip"), str(tmpdir))
    subprocess.run(["unzip", "US_county.zip"], cwd=str(tmpdir), check=True)

    return pathlib.Path(str(tmpdir))
