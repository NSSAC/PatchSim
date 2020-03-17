"""Tests from the manual tests directory."""

from util import assert_equal_files, chdir_context

import patchsim as sim


def test_det(workdir):
    """Run the test_det setting."""
    with chdir_context(workdir):
        configs = sim.read_config("cfg_test_det")
        sim.run_disease_simulation(configs, write_epi=True)

        assert_equal_files("test_det.out", "test_det.out.expected")


def test_foi(workdir):
    """Run the test_foi setting."""
    with chdir_context(workdir):
        configs = sim.read_config("cfg_test_foi")
        sim.run_disease_simulation(configs, write_epi=True)

        assert_equal_files("test_foi.out", "test_foi.out.expected")


def test_paramfile(workdir):
    """Run the test_paramfile setting."""
    with chdir_context(workdir):
        configs = sim.read_config("cfg_paramfile")
        sim.run_disease_simulation(configs, write_epi=True)

        assert_equal_files("test.out", "test_paramfile.out.expected")


def test_stoc(workdir):
    """Run the test_stoc setting."""
    with chdir_context(workdir):
        configs = sim.read_config("cfg_test_stoc")
        sim.run_disease_simulation(configs, write_epi=True)

        assert_equal_files("test_stoc.out", "test_stoc.out.expected")
