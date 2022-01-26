__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/09/03 09:36:00"

import pickle
from math import sqrt
from random import randrange, sample

from analyzer import Highlighter, AncestorsHandler
from VAE_accessor import VAEAccessor
from download_MSA import Downloader
from experiment_handler import ExperimentStatistics
from msa_preparation import MSA
from parser_handler import CmdHandler
from sequence_transformer import Transformer


class MutagenesisGenerator:
    def __init__(self, setuper, ref_dict=None, num_point_mut=1, distance_threshold=0.2):
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

    def reconstruct_ancestors(self, samples=10):
        mutants_samples = []
        ancestors = []
        anc_names = []
        anc_n = self.cur_name

        ancestor_dist, ancestor, anc_pos, _ = self._get_ancestor(mutants=self.cur)
        ancestors.append(anc_pos)
        anc_names.append(anc_n)
        while ancestor_dist > self.threshold:
            self.anc_seqs.append(ancestor)
            mutant_seqs = self._produce_mutants(ancestor, samples)
            anc_n = 'anc_{}'.format(len(mutants_samples))
            ancestor_dist, ancestor, anc_pos, mutants_pos = self._get_ancestor(mutants=mutant_seqs)
            mutants_samples.append(mutants_pos)
            ancestors.append(anc_pos)
            anc_names.append(anc_n)
        return ancestors, anc_names, mutants_samples

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

    def _mutants_positions(self, seqs):
        binary, weights, keys = self.transformer.prepare_aligned_msa_for_vae(seqs)
        data, _ = self.handler.propagate_through_VAE(binary, weights, keys)
        return data

    def _get_ancestor(self, mutants):
        mutants_pos = self._mutants_positions(mutants)
        seqs = list(mutants.values())
        min_dist = float("inf")
        ancestor = []
        anc_pos = []
        for i, p in enumerate(mutants_pos):
            dis = sqrt(p[0] ** 2 + p[1] ** 2)
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
        ref_pos = self._mutants_positions(self.cur)
        coor_tuple = tuple(ref_pos[0])
        tuple_dim = range(len(ref_pos[0]))
        step_list = [0.0 for _ in tuple_dim]
        for i in tuple_dim:
            step_list[i] = (coor_tuple[i] / cnt_of_anc)
        i = 30
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
        # to_highlight = successors
        ancestors_to_store = self.handler.decode_z_to_aa_dict(successors, self.cur_name)
        file_name = '{}straight_ancestors.fasta'.format(self.setuper.model_name)
        observing_probs = self.exp_handler.create_and_store_ancestor_statistics(ancestors_to_store, file_name,
                                                                                coords=successors)
        return list(ancestors_to_store.keys()), to_highlight, observing_probs

    def measure_probability_given_ancestors(self):
        """ Measure probability of ancestors given in file """
        if self.setuper.align:
            ancestors = MSA.load_msa(file=self.setuper.highlight_files)
            ancestors = AncestorsHandler(setuper=self.setuper, seq_to_align=ancestors).align_to_ref()
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


if __name__ == '__main__':
    tar_dir = CmdHandler()
    dow = Downloader(tar_dir)

    # ref = MutagenesisGenerator.binary_to_seq(tar_dir,seq_key='P27652.1')
    # ref_binary = MutagenesisGenerator.binary_to_seq(tar_dir, seq_key='P59336_S14', return_binary=True)
    mut = MutagenesisGenerator(setuper=tar_dir, num_point_mut=tar_dir.mut_points)
    # ancs, names, muts = mut.reconstruct_ancestors(samples=tar_dir.mutant_samples)
    # h = Highlighter(tar_dir)
    # h.highlight_mutants(ancs, names, muts, file_name='mutans_{}_{}'.format(tar_dir.mut_points, tar_dir.mutant_samples))
    # if tar_dir.focus:
    #     h.highlight_mutants(ancs, names, muts, file_name='mutans_focus_{}_{}'.format(tar_dir.mut_points, tar_dir.mutant_samples), focus=tar_dir.focus)
    # mut.store_ancestors_in_fasta(names)
    names, ancestors, probs = mut.get_straight_ancestors()
    # given_anc_probs, given_anc_names = mut.measure_probability_given_ancestors()
    # h.highlight_mutants(ancs=ancestors, names=names, mutants=[], file_name='straight_ancestors_no_focus', focus=False)
    h = Highlighter(tar_dir)
    h.plot_probabilities(probs, ancestors)
    # h.plot_straight_probs_against_ancestors(probs, given_anc_probs, given_anc_names)
