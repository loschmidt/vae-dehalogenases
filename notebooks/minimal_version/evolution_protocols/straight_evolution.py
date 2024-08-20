"""
Ancestors generator for VAEncestors
Generate 100 ancestors from latent space along the trajectory from evo_query embedding to latent space origin
"""
__author__ = "Pavel Kohout <pavel.kohout@rexetox.muni.cz>"
__date__ = "2024/08/14 09:36:00"

import sys

import numpy as np
import torch

# My libraries
from notebooks.minimal_version.evolution_protocols.profiler import Profiler
from notebooks.minimal_version.latent_space import LatentSpace
from notebooks.minimal_version.msa import MSA
from notebooks.minimal_version.parser_handler import RunSetup


def generate_regular_trajectory(query, steps):
    """
    Generate
    :param query: 
    :param steps: 
    :return:
    """
    # Get the dimensionality of the query point
    dim = len(query)

    # Create an array of zeros to store the generated points
    points = np.zeros((steps + 1, dim))

    # Generate points in regular steps from query to 0
    for i in range(steps + 1):
        points[i] = query - (query / steps) * i

    return points.astype(np.float32)


class StraightEvolution:
    def __init__(self, run: RunSetup):
        self.ancestors = run.ancestors
        self.latent_space = LatentSpace(run)
        self.vae = self.latent_space.vae
        self.conditional = self.latent_space.conditional
        self.pickle = run.pickles
        self.run = run
        self.anc_seqs = []
        self.profiler = Profiler(run)

    def get_ancestors(self, profile: bool = True, c: torch.Tensor = None):
        """
         Method does that the line between a position of evo_query sequence
         and the center (0,0) is divided to ancestors intervals and their
         border are sampled for the reconstruction. It can be used as a validation
         metric that int the randomly initialized weights od encoder has no
         effect on latent space if the radiuses of differently init models
         are almost same.
         :param profile: should generate profile
         :param c: conditional label to be applied to all ancestors
         :return: ancestor dictionary
        """
        print(" Straight evolution protocol : Straight ancestors generating process started")

        if c is None and self.latent_space.conditional:
            print("Please provide label for your ancestors")
            sys.exit(1)

        query_emb = self.latent_space.key_to_embedding(self.run.evo_query)
        points = torch.from_numpy(generate_regular_trajectory(query_emb, self.ancestors))
        # ancestors_to_store = self.latent_space.decode_z_to_aa_dict(points)
        if c is not None:
            c = c.unsqueeze(0).repeat(points.shape[0], 1)
        with torch.no_grad():
            number_sequences = self.vae.z_to_number_sequences(points, c).cpu().numpy()
            ancestor_dict_num = {f"anc{i}": seq for i, seq in enumerate(number_sequences)}
            ancestors_to_store = MSA.number_to_amino(ancestor_dict_num)
        print(ancestors_to_store)
        file_name = 'straight_latent_ancestors.fasta'
        if profile:
            self.profiler.profile_sequences(ancestors_to_store, file_name, points.cpu().numpy())
        return ancestors_to_store
