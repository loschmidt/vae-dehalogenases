""" Compute and plot solubility distribution and bins with given thresholds"""
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

LOW = 0.25
MED = 0.6


def solubility_dist(file, keys):
    solubilities = []
    with open(file, "r") as sol_data:
        for i, record in enumerate(sol_data):
            if i == 0:  # header
                continue
            key, sol = record.split("\t")
            if key in keys:
                solubilities.append(sol)
        float_solubilities = np.array(solubilities).astype(float)
        solubility_bins = []
        # include to bins
        for sd in float_solubilities:
            if sd < LOW:
                solubility_bins.append(0)
            elif sd <= MED:
                solubility_bins.append(1)
            else:
                solubility_bins.append(2)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        n, bins, patches = ax1.hist(float_solubilities, 50, alpha=0.75)
        ax1.set_xtitle = 'Solubility'
        ax1.set_ytitle = 'Frequencies'

        n, bins, patches = ax2.hist(solubility_bins, 3)
        ax2.set_xtitle = f'Solubility bins (LOW <{LOW}<= MEDIUM <={MED}< HIGH)'
        ax2.set_ytitle = 'Frequencies'
        fig.savefig('sol_dist.png')


# Program parse ready
parser = argparse.ArgumentParser(description='Script for solubility distributions')
parser.add_argument("--file", help="Solubility file")
parser.add_argument("--keys", help="Key list pickle")

# Run script
args = parser.parse_args()

with open(args.keys, 'rb') as file_handle:
    keys = pickle.load(file_handle)

solubility_dist(args.file, keys)