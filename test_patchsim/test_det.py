"""Test Patchsim."""

import os
import pathlib
import shutil
import filecmp

from pytest import fixture

import patchsim as sim

@fixture
def workdir(tmpdir):
    """Create a working directory PatchSim code."""
    basedir = pathlib.Path(__file__).resolve().parents[1] / "manual_tests"

    fnames = [
        "cfg_test_det",
        "test_pop.txt",
        "test_net.txt",
        "test_seed.txt",
        "test_vax.txt",
        "test_det.out.expected",
    ]

    for fname in fnames:
        shutil.copy(str(basedir / fname), str(tmpdir))

    return pathlib.Path(str(tmpdir))


def test_det(workdir):
    """Run the test_det script."""
    curdir = os.getcwd()
    try:
        os.chdir(str(workdir))
        configs = sim.read_config("cfg_test_det")
        sim.run_disease_simulation(configs, write_epi=True)

        assert filecmp.cmp(workdir / "test_det.out", workdir / "test_det.out.expected", shallow=False)
    finally:
        os.chdir(curdir)
