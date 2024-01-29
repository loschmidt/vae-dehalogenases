__author__ = "Pavel Kohout <xkohou15@vutbr.cz>"
__date__ = "2024/01/29 09:36:00"

import pickle

import os
import sys
import inspect
from typing import Optional

import numpy as np

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from VAE_accessor import VAEAccessor
from experiment_handler import ExperimentStatistics
from msa_handlers.msa_preparation import MSA
from parser_handler import CmdHandler


class GridGenerator:
    grid_size = 60

    def __init__(self, setuper):
        self.handler = VAEAccessor(setuper, model_name=setuper.get_model_to_load())
        self.pickle = setuper.pickles_fld
        self.setuper = setuper

        self.aa, self.aa_index = MSA.aa, MSA.amino_acid_dict(self.pickle)
        self.exp_handler = ExperimentStatistics(setuper, experiment_name="Grid sampler")

    def get_grid_samples(self, grid_size=60):
        """
        Generate grid sequences
        """
        grid_start = None

        def create_regular_grid(density: int, latent_start: Optional[int], dim: int) -> np.ndarray:
            latent_start = latent_start if latent_start is not None else 6
            start = min(latent_start, -1 * latent_start)
            end = max(latent_start, -1 * latent_start)
            step = (end - start) / (density - 1)

            grids = [np.arange(start, end + step, step) for _ in range(dim)]
            meshes = np.meshgrid(*grids, indexing='ij')
            points = np.column_stack([mesh.flatten() for mesh in meshes])
            return points

        grid_points = create_regular_grid(grid_size, grid_start, 2)
        start_pr = [grid_start if grid_start is not None else 6 for _ in range(2)]
        print(f"Grid points generated: {grid_points.shape}, per row {grid_size}, and grid start ({start_pr})")

        grid_samples = self.handler.decode_z_to_aa_dict(grid_points, "init", 0)
        grid_fasta = {}
        for i, s in enumerate(grid_samples.values()):
            grid_fasta[f"seq_{i}"] = s

        name_grid_coords = {}
        for seq_name, point in zip(grid_fasta.keys(), grid_points):
            name_grid_coords[seq_name] = point

        file_name = 'grid.fasta'
        self.exp_handler.store_ancestor_dict_in_fasta(seq_dict=grid_fasta, file_name=file_name)

        path_grid_pkl = os.path.join(self.pickle, "grid_name_to_point.pkl")
        print(f"Storing fasta file in {file_name} and mapping to coords in {path_grid_pkl}")
        with open(path_grid_pkl, "wb") as f:
            pickle.dump(name_grid_coords, f)


def run_grid_generation():
    """
    Generate grid with sequences
    """
    cmd_line = CmdHandler()
    grid = GridGenerator(setuper=cmd_line)
    grid.get_grid_samples()
