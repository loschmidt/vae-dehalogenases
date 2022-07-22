__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/07/22 10:36:00"

from msa_handlers.msa_preparation import MSA
from project_enums import SolubilitySetting
from parser_handler import CmdHandler
from VAE_accessor import VAEAccessor
from Statistics.entropy_latent import EntropyLatent
from reconstruction.curves.curves import DiscreteCurve
from reconstruction.mutagenesis import MutagenesisGenerator
from analyzer import Highlighter
from experiment_handler import ExperimentStatistics

from copy import deepcopy

import torch
import matplotlib.pyplot as plt


class Riemannian:
    """
    Using Riemannian trajectories on manifold
    """

    def __init__(self, cmd_handler: CmdHandler):
        self.vae = VAEAccessor(cmd_handler, model_name=cmd_handler.get_model_to_load())
        self.cmd = cmd_handler
        self.entropy_latent, self.distnet = self.init_entropy()

    def init_entropy(self):
        """ Prepare entropy fit entropy noise """
        entropy_latent = EntropyLatent(self.cmd, optimized_entropy=True)

        # calculate entropy and create space
        entropy_latent.create_plot()

        distnet = entropy_latent.distnet
        return entropy_latent, distnet

    def add_entropy_noise(self, z, probs):
        """ Add Entropy noise wile reconstructing """
        s = self.distnet(z).view(*z.shape[:-1], 1, 1)
        probs = (1 - s) * probs + s * probs.mean()
        return probs

    def entropy_decoder(self, z):
        """ Sample z from the latent space and apply entropy noise """
        solubility = self.cmd.get_solubility_data()

        if solubility is not None:
            solubility = torch.tensor([SolubilitySetting.SOL_BIN_MEDIUM.value for _ in range(z.shape[0])])

        final_shape = (z.shape[0], -1, len(MSA.aa) + 1)
        log_p = self.vae.vae.decoder(z, c=solubility)
        probs = torch.unsqueeze(log_p, -1)
        probs = probs.view(final_shape)
        probs = torch.exp(probs)

        probs = self.add_entropy_noise(z, probs)
        return probs

    def curve_energy(self, curve):
        """ get energy of curve, as described in https://doi.org/10.1038/s41467-022-29443-w """
        if curve.dim() == 2: curve.unsqueeze_(0)  # BxNxd

        recon = self.entropy_decoder(curve)  # BxNxFxS

        x = recon[:, :-1, :, :]
        y = recon[:, 1:, :, :]  # Bx(N-1)xFxS
        dt = torch.norm(curve[:, :-1, :] - curve[:, 1:, :], p=2, dim=-1)  # Bx(N-1)
        energy = (1 - (x * y).sum(dim=2)).sum(dim=-1)  # Bx(N-1)
        return 2 * (energy * dt).sum(dim=-1)

    def curve_length(self, curve):
        """ as described in https://doi.org/10.1038/s41467-022-29443-w"""
        return torch.sqrt(self.curve_energy(curve))

    def numeric_curve_optimizer(self, curve):
        optimizer = torch.optim.Adam([curve.parameters], lr=1e-2)
        alpha = torch.linspace(0, 1, 50).reshape((-1, 1))
        best_curve, best_loss = deepcopy(curve), float('inf')
        for i in range(10):
            optimizer.zero_grad()
            loss = self.curve_energy(curve(alpha)).sum()
            loss.backward()
            optimizer.step()
            grad_size = torch.max(torch.abs(curve.parameters.grad))
            if grad_size < 1e-3:
                break
            if loss.item() < best_loss:
                best_curve = deepcopy(curve)
                best_loss = loss.item()

        return best_curve


def sequence_encoding(sequence, cmd):
    """ get encoding of sequence in the latent space, if None query sequence is chosen """
    sequence = sequence if sequence else cmd.query_id

    mut = MutagenesisGenerator(setuper=cmd)
    sequence_dict = mut.transformer.get_seq_dict_by_key(sequence)
    return mut.mutants_positions(sequence_dict)[0]


def run_riemannian_ancestors(sequence=None):
    """ Reconstruct manifold path to the center from sequence """
    cmd_line = CmdHandler()
    riemannian = Riemannian(cmd_handler=cmd_line)

    x2 = torch.tensor([0.0, 0.0])
    x1 = torch.tensor(sequence_encoding(sequence, cmd_line))
    curve = DiscreteCurve(x1, x2)

    curve = riemannian.numeric_curve_optimizer(curve)
    print(curve.parameters)
    trajectory = curve(torch.linspace(0, 1, cmd_line.ancestral_samples + 1)).detach()

    # plot and create report
    fig, ax = plt.subplots(constrained_layout=True)
    riemannian.entropy_latent.create_plot(ax)
    ax.scatter(trajectory[:, 0], trajectory[:, 1], s=1, c="#985C2F")
    fig.savefig(cmd_line.high_fld + "/riemannian_ancestors.png", dpi=600)

    experiment = ExperimentStatistics(cmd_line, experiment_name="riemannian_ancestors")
    ancestors_to_store = riemannian.vae.decode_z_to_aa_dict(trajectory, cmd_line.query_id)
    observing_probs = experiment.create_and_store_ancestor_statistics(ancestors_to_store, "riemannian_ancestors",
                                                                      coords=trajectory)

    h = Highlighter(cmd_line)
    h.plot_probabilities(observing_probs, trajectory)
