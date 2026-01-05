# General functions for the statistical analysis of the simulation results
import numpy as np
import glob
import os

# function to load results file
def load_results(path):
    data = np.load(path, allow_pickle = True)
    phi = data['phi'] # final state phi per iteration
    types = data['types'] # final state list of types per iteration, corresponding to the order of phi
    death_log = data['death_log'] # every cell that died and its type
    division_log = data['division_log'] # idem, for division
    params = data['params'] # dictionary of all parameters used in the simulation
    return phi, types, death_log, division_log, params


# function to get the lowest y-coordinate of cell type 1
def lowest_y(phi, types):
    lowest = None
    for k,type in enumerate(types):
        if type == 1:
            for x in range(phi.shape[0]):
                for y in range(phi.shape[1]):
                    if phi[x,y,k] > 0.5:
                        if lowest is None or y < lowest:
                            lowest = y
    return lowest


# use function h for volumes
def h(x):
    return x**2*(3-2*x)


# function to compute the volume of cancer cells and healthy cells
def volumes(phi, types):
    volumes = {1: [], 2:[]}
    for k, type in enumerate(types):
        if type in (1,2): # substrate is irrelevant
            vol = np.sum(h(phi[:,:,k]))
            volumes[type].append(vol)
    return volumes


# function to count the number of deaths per cell types
def count_deaths(death_log):
    counts = { 1: 0, 2: 0}
    for type in death_log:
        counts[type] += 1
    return counts


# function to get results per experiment
def analyze_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    results_summary = {}

    for f in files:
        phi_list, types_list, death_logs, _, _ = load_results(f)

        trial_stats = []
        for phi, types, death_log in zip(phi_list, types_list, death_logs):
            stats = {
                "lowest_y": lowest_y(phi, types),
                "volumes": volumes(phi, types),
                "deaths": count_deaths(death_log)
            }
            trial_stats.append(stats)

        results_summary[os.path.basename(f)] = trial_stats

    return results_summary

