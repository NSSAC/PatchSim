import sys
sys.path.append('../')
import patchsim as sim
configs = sim.read_config('cfg_test_det')
sim.run_disease_simulation(configs,write_epi=True)
