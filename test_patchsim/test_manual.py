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


def test_stopstart(workdir):
    """Run the test_stopstart setting."""
    with chdir_context(workdir):
        configs = sim.read_config("cfg_check1")
        sim.run_disease_simulation(configs, write_epi=True)
        configs = sim.read_config("cfg_check2")
        sim.run_disease_simulation(configs, write_epi=True)
        configs = sim.read_config("cfg_check3")
        sim.run_disease_simulation(configs, write_epi=True)

        assert_equal_files("test1.out", "test1_stopstart.out.expected")
        assert_equal_files("test2.out", "test2_stopstart.out.expected")
        assert_equal_files("test3.out", "test3_stopstart.out.expected")
