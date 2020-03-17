import sys
sys.path.append('../')
import patchsim as sim

configs = sim.read_config('cfg_check1')
sim.run_disease_simulation(configs,write_epi=True)
configs = sim.read_config('cfg_check2')
sim.run_disease_simulation(configs,write_epi=True)
configs = sim.read_config('cfg_check3')
sim.run_disease_simulation(configs,write_epi=True)
