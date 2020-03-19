import sys
sys.path.append('../')
import patchsim as sim

from fnmatch import fnmatch

class NetIntervention:
    def __init__(self, configs):
        self.patch_idx = None
        self.intervention_file = configs["NetInterventionFile"]

        self.src_glob = [] # array of strings
        self.dst_glob = [] # array of strings
        self.src_idxs = [] # array of array of strings
        self.dst_idxs = [] # array of array of strings
        self.T_starts = [] # array of ints
        self.T_ends = [] # array of ints
        self.fractions = [] # array of floats

        self.original_Theta = None

    def populate(self):
        with open(self.intervention_file, "r") as fobj:
            for line in fobj:
                T_start, T_end, src_glob, dst_glob, fraction = line.strip().split(",")
                T_start, T_end = int(T_start), int(T_end)
                fraction = float(fraction)
                src_idxs = sorted(self.patch_idx[id_] for id_ in self.patch_idx if fnmatch(id_, src_glob))
                dst_idxs = sorted(self.patch_idx[id_] for id_ in self.patch_idx if fnmatch(id_, dst_glob))

                self.src_glob.append(src_glob)
                self.dst_glob.append(dst_glob)
                self.src_idxs.append(src_idxs)
                self.dst_idxs.append(dst_idxs)
                self.T_starts.append(T_start)
                self.T_ends.append(T_end)
                self.fractions.append(fraction)

    def __call__(self, configs, patch_df, params, Theta, seeds, vaxs, t):
        if self.patch_idx is None:
            self.patch_idx = {id_: i for i, id_ in enumerate(patch_df["id"])}
            self.populate()
            self.original_Theta = Theta.copy()

        Theta[:,:,:] = self.original_Theta
        for src_idxs, dst_idxs, T_start, T_end, fraction in zip(self.src_idxs, self.dst_idxs, self.T_starts, self.T_ends, self.fractions):
            if T_start <= t < T_end:
                for src_id in src_idxs:
                    for dst_id in dst_idxs:
                        Theta[:, src_id, dst_id] *= fraction



configs = sim.read_config('cfg_test_foi')
my_intervention = NetIntervention(configs)
sim.run_disease_simulation(configs, write_epi=True, intervene_step=my_intervention)

# As for intervention specification, perhaps a format like below will help:
# T_start, T_end, src_regex, dest_regex, fraction

# T's refer to tick of the simulation.
# The regex entries will look like: 51*
# (all counties of Virginia, FIPS code 51) or *s (all school children).
# So 51*s will be all school children in Virginia.
# This way we can specify which subset of edges to tune up or down.
# The +/- column will say if it is to be scaled up or down,
# and percentage column will tell the relative proportion
# w.r.t value at T_start.
