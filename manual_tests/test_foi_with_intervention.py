import sys
sys.path.append('../')
import patchsim as sim

def my_intervention(configs, patch_df, params, Theta, seeds, vaxs, t):
    beta = params["beta"]
    beta[:,:] = beta * 0.9

configs = sim.read_config('cfg_test_foi')
sim.run_disease_simulation(configs, write_epi=True, intervene_step=my_intervention)
