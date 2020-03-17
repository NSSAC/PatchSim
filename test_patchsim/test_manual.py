"""Tests from the manual tests directory."""

from util import assert_equal_files, chdir_context

import patchsim as sim

def test_det(workdir):
    """Run the test_det script."""
    with chdir_context(workdir):
        configs = sim.read_config("cfg_test_det")
        sim.run_disease_simulation(configs, write_epi=True)

        assert_equal_files("test_det.out", "test_det.out.expected")
