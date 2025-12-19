import numpy as np

# Example of loading saved data
data = np.load("results/division_experiment/division_rate_0.0004.npz", allow_pickle=True)
# Attributes available: phi, types, death_log, division_log, params
phi = data['phi'] # final state phi per iteration
types = data['types'] # final state list of types per iteration, corresponding to the order of phi
death_log = data['death_log'] # every cell that died and its type
division_log = data['division_log'] # idem, for division
params = data['params'] # dictionary of all parameters used in the simulation

