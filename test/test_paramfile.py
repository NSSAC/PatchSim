import sys
sys.path.append('../')
import patchsim as sim
configs = sim.read_config('cfg_paramfile')
sim.run_disease_simulation(configs,write_epi=True)
