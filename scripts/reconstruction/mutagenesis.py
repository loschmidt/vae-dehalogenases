__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/09/03 09:36:00"

import pickle
import torch
import numpy as np
from math import sqrt
from random import randrange, sample

import os
import sys
import inspect

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from analyzer import Highlighter, AncestorsHandler
from VAE_accessor import VAEAccessor
from msa_handlers.download_MSA import Downloader
from experiment_handler import ExperimentStatistics
from msa_handlers.msa_preparation import MSA
from parser_handler import CmdHandler
from sequence_transformer import Transformer


class MutagenesisGenerator:

    number_of_successors = 3

    def __init__(self, setuper, ref_dict=None, num_point_mut=1, distance_threshold=0.08):
        self.num_mutations = num_point_mut
        self.handler = VAEAccessor(setuper, model_name=setuper.get_model_to_load())
        self.threshold = distance_threshold
        self.pickle = setuper.pickles_fld
        self.setuper = setuper
        self.anc_seqs = []
        self.transformer = Transformer(setuper)

        self.aa, self.aa_index = MSA.aa, MSA.amino_acid_dict(self.pickle)

        if ref_dict is None:
            with open(self.pickle + "/reference_seq.pkl", 'rb') as file_handle:
                ref_dict = pickle.load(file_handle)
        self.cur = ref_dict
        self.cur_name = list(ref_dict.keys())[0]
        self.exp_handler = ExperimentStatistics(setuper, experiment_name="Mutagenesis_experiment")

    def random_mutagenesis(self, samples=10, multicriterial=False):
        """ Simulate random walk in the latent space by sequence mutation """
        mutants_coords = []
        ancestors_coords, ancestors_seq = [], []
        anc_names = []
        anc_n = self.cur_name

        ancestor_dist, ancestor, anc_pos, _ = self.get_closest_mutant(mutants=self.cur)
        ancestors_coords.append(anc_pos)
        ancestors_seq.append(list(self.cur.values())[0])
        anc_names.append(anc_n)
        while ancestor_dist > self.threshold:
            self.anc_seqs.append(ancestor)
            mutant_seqs = self._produce_mutants(ancestor, samples)
            anc_n = 'anc_{}'.format(len(mutants_coords))
            ancestor_dist, ancestor, anc_pos, mutants_pos = self.get_closest_mutant(mutants=mutant_seqs)
            # Multi-criterial optimization with sequence likelihood
            if multicriterial:
                anc_pos, ancestor = self.pareto_optimalization(mutant_seqs, mutants_pos)
            mutants_coords.append(mutants_pos)
            ancestors_coords.append(anc_pos)
            ancestors_seq.append(ancestor)
            anc_names.append(anc_n)
        return ancestors_coords, anc_names, mutants_coords, ancestors_seq

    def pareto_optimalization(self, mutants_seq, mutants_pos):
        """ Use multicriterial selection of sequence coords and likelihood """
        seq_binaries, _, keys = self.transformer.sequence_dict_to_binary(mutants_seq)

        # return log likelihood which should be maximized thus exponential and 1 minus exp
        # then minimization for either distance or likelihood
        seq_binaries = torch.from_numpy(seq_binaries)
        log_like = 1 - torch.exp(self.handler.get_marginal_probability(seq_binaries, multiple_likelihoods=True))
        # sequence distance normalized
        seq_distances = np.array([np.sqrt(d.dot(d)) for d in mutants_pos])
        seq_distances_norm = seq_distances / seq_distances.max()

        # pareto selection simplified by proximity in 2D graph to optimal solution at the origin
        min_idx, min_val = 0, 60000
        for i, (like, d) in enumerate(zip(log_like, seq_distances_norm)):
            v = np.array([like*1000, d]) # more significant likelihood
            v_dist = np.sqrt(v.dot(v))
            if v_dist < min_val:
                min_val = v_dist
                min_idx = i
        return mutants_pos[min_idx], mutants_seq["mut_{}".format(min_idx)]

    def _produce_mutants(self, gene, samples):
        mutant_seqs = {}  # List of dictionaries with fake names
        # Not automatizated, user has to fing cat residues on his own
        # Process:
        #       use create_fasta_file method to generate fasta MSA
        #       upload it to ConSurf Server and let proceed it
        cat_positions = []  # Positions of catalytic residues

        def random_mutation(c):
            m = self.aa[randrange(len(self.aa))]
            while m == c:
                m = self.aa[randrange(len(self.aa))]
            return m

        if self.num_mutations > len(gene):
            print('Error occured, number of mutations is bigger than length of gene', self.num_mutations, len(gene))
            exit(1)
        for i in range(samples):
            # Select position for mutations
            pos = sample(range(len(gene)), self.num_mutations)
            for ii in range(len(pos)):
                if pos[ii] in cat_positions:
                    pos.pop(ii)  ## Not robust but catalytic residues wont be used in this case anyway
            mutant = gene.copy()
            for p in pos:
                mutant[p] = random_mutation(mutant[p])
            mutant_seqs['mut_' + str(i)] = mutant

        return mutant_seqs

    def mutants_positions(self, seqs):
        """ Get mutants position in the latent space from dictionary """
        binary, weights, keys = self.transformer.sequence_dict_to_binary(seqs)
        data, _ = self.handler.propagate_through_VAE(binary, weights, keys)
        return data

    def get_closest_mutant(self, mutants):
        """ Get mutant with highest proximity to the origin """
        mutants_pos = self.mutants_positions(mutants)
        seqs = list(mutants.values())
        min_dist = float("inf")
        ancestor = []
        anc_pos = []
        for i, p in enumerate(mutants_pos):
            dis = sqrt(sum([p_i**2 for p_i in p]))#sqrt(p[0] ** 2 + p[1] ** 2)
            if dis < min_dist:
                min_dist = dis
                ancestor = seqs[i]
                anc_pos = p

        return min_dist, ancestor, anc_pos, mutants_pos

    def get_straight_ancestors(self, cnt_of_anc=100):
        """
         Method does that the line between a position of reference sequence
         and the center (0,0) is divided to cnt_of_anc and the borders of the
         intervals are denoted as ancestors. It can be used as a validation
         metric that int the randomly initialized weights od encoder has no
         effect on latent space if the radiuses of differently init models
         are almost same.
         Also reconstructs 30 successor sequences.
         The class returns only ancestors!!!
        """
        print(" Mutagenesis message : Straight ancestors generating process started")
        ref_pos = self.mutants_positions(self.cur)
        coor_tuple = tuple(ref_pos[0])
        tuple_dim = range(len(ref_pos[0]))
        step_list = [0.0 for _ in tuple_dim]
        for i in tuple_dim:
            step_list[i] = (coor_tuple[i] / cnt_of_anc)
        i = MutagenesisGenerator.number_of_successors
        successors = []
        while i > 0:
            cur_coor = [0.0 for _ in tuple_dim]
            for s_i in tuple_dim:
                cur_coor[s_i] = coor_tuple[s_i] + (step_list[s_i] * i)
            successors.append(tuple(cur_coor))
            i -= 1
        i = 1
        to_highlight = [coor_tuple]
        while i <= cnt_of_anc:
            cur_coor = [0.0 for _ in tuple_dim]
            for s_i in tuple_dim:
                cur_coor[s_i] = coor_tuple[s_i] - (step_list[s_i] * i)
            to_highlight.append(tuple(cur_coor))
            i += 1

        successors.extend(to_highlight)
        ancestors_to_store = self.handler.decode_z_to_aa_dict(successors, self.cur_name,
                                                              MutagenesisGenerator.number_of_successors)
        file_name = "ancestors_straight_evolutionary_strategy.fasta"
        observing_probs = self.exp_handler.create_and_store_ancestor_statistics(ancestors_to_store, file_name,
                                                                                coords=successors)
        return list(ancestors_to_store.keys()), to_highlight, observing_probs

    def measure_probability_given_ancestors(self):
        """ Measure probability of ancestors given in file """
        if self.setuper.align:
            ancestors = MSA.load_msa(file=self.setuper.highlight_files)
            ancestors = AncestorsHandler(setuper=self.setuper).align_to_ref(ancestors)
            # probs = ProbabilityMaker(None, None, self.setuper, generate_negative=False).measure_seq_probability(
            #     ancestors)
            file_name = '{}sebestova_ancestors.fasta'.format(self.setuper.model_name)
            # self._store_in_fasta_csv(ancestors, to_file=file_name, probs=probs, )
            observing_probs = self.exp_handler.create_and_store_ancestor_statistics(ancestors, file_name,
                                                                                 coords=[(0.0, 0.0) for _ in ancestors])
            return observing_probs, list(ancestors.keys())
        else:
            print(" Mutagenesis message : Ancestor's probabilities measurement - not given file and align option ")
            return [], []


def run_random_mutagenesis(multicriterial=False):
    """
    Task description for random mutagenesis approach
    If the multicriterial parameters is on than we are doing multi-criterial optimization with likelihood of sequence
    """
    cmd_line = CmdHandler()
    mutation_generator = MutagenesisGenerator(setuper=cmd_line, num_point_mut=cmd_line.mut_points)
    anc_coords, anc_names, mutants_coords, anc_seq = \
        mutation_generator.random_mutagenesis(samples=cmd_line.mutant_samples, multicriterial=multicriterial)

    file_templ = 'random{}_' + str(cmd_line.mut_points) + '_points_mutants_' + str(cmd_line.mutant_samples)
    file_templ = file_templ + "_multi" if multicriterial else file_templ
    h = Highlighter(cmd_line)
    h.highlight_mutants(anc_coords, anc_names, mutants_coords,
                        file_name=file_templ.format(''))
    if cmd_line.focus:
        h.highlight_mutants(anc_coords, anc_names, mutants_coords, file_name=file_templ.format('_focus'),
                            focus=cmd_line.focus)

    # prepare sequence dictionary
    seq_dict = {}
    for seq_name, seq in zip(anc_names, anc_seq):
        seq_dict[seq_name] = seq

    mutation_generator.exp_handler.create_and_store_ancestor_statistics(seq_dict, file_templ.format(''), anc_coords)


def run_straight_evolution():
    """
    Task description for straight ancestral reconstruction strategy
    """
    cmd_line = CmdHandler()

    # Stats are created inside get_straight_ancestors method
    mut = MutagenesisGenerator(setuper=cmd_line, num_point_mut=cmd_line.mut_points)
    names, ancestors, probs = mut.get_straight_ancestors(cmd_line.ancestral_samples)

    h = Highlighter(cmd_line)
    h.plot_probabilities(probs, ancestors)


if __name__ == '__main__':
    tar_dir = CmdHandler()
    dow = Downloader(tar_dir)

    mut = MutagenesisGenerator(setuper=tar_dir, num_point_mut=tar_dir.mut_points)
    names, ancestors, probs = mut.get_straight_ancestors()
    given_anc_probs, given_anc_names = mut.measure_probability_given_ancestors()
    # h.highlight_mutants(ancs=ancestors, names=names, mutants=[], file_name='straight_ancestors_no_focus', focus=False)
    h = Highlighter(tar_dir)
    h.plot_probabilities(probs, ancestors)
    h.plot_straight_probs_against_ancestors(probs, given_anc_probs, given_anc_names)
