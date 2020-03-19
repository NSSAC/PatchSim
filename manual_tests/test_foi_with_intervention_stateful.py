import sys
sys.path.append('../')
import patchsim as sim

class MyIntervention:
    def __init__(self):
        self.original_beta = None
        self.reduced_beta = None

    def __call__(self, configs, patch_df, params, Theta, seeds, vaxs, t):
        if self.original_beta is None:
            self.original_beta = params["beta"]
            self.reduced_beta = params["beta"] * 0.9

        # For first 10 step do nothing to beta
        if t < 10:
            params["beta"] = self.original_beta
        # Next 10 steps keep it low
        elif 10 <= t < 20:
            params["beta"] = self.reduced_beta
        # After that restore
        else:
            params["beta"] = self.original_beta

configs = sim.read_config('cfg_test_foi')
my_intervention = MyIntervention()

sim.run_disease_simulation(configs, write_epi=True, intervene_step=my_intervention)
