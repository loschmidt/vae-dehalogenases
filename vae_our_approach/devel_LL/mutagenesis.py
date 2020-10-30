__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/09/03 09:36:00"

from analyzer import VAEHandler, Highlighter
from msa_filter_scorer import MSAFilterCutOff as Convertor
from msa_prepar import MSA
from pipeline import StructChecker
from download_MSA import Downloader
from random import randrange, sample
from math import sqrt
from benchmark import Benchmarker as ProbabilityMaker

import numpy as np
import pickle
import csv
import os.path as path
from scipy.stats import multivariate_normal as norm

class MutagenesisGenerator:
    def __init__(self, setuper, ref_dict=None, num_point_mut=1, distance_threshold=0.2):
        self.num_mutations = num_point_mut
        self.handler = VAEHandler(setuper)
        self.threshold = distance_threshold
        self.pickle = setuper.pickles_fld
        self.setuper = setuper
        self.anc_seqs = []

        self.msa_obj = MSA(setuper=self.setuper, processMSA=False)
        self.aa, self.aa_index = self.msa_obj.amino_acid_dict(export=True)

        if ref_dict is None:
            with open(self.pickle + "/reference_seq.pkl", 'rb') as file_handle:
                ref_dict = pickle.load(file_handle)
        self.cur = ref_dict
        self.cur_name = list(ref_dict.keys())[0]

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
        cat_positions = [] # Positions of catalytic residues

        def random_mutation(c):
            m = self.aa[randrange(len(self.aa))]
            while  m == c:
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
                    pos.pop(ii) ## Not robust but catalytic residues wont be used in this case anyway
            mutant = gene.copy()
            for p in pos:
                mutant[p] = random_mutation(mutant[p])
            mutant_seqs['mut_'+str(i)] = mutant

        return mutant_seqs

    def check_mutant(self,  num_m, gene=['A','A','A','A','A','A','A','A','A','A']):
        self.num_mutations = num_m
        muts = self._produce_mutants(gene, samples=1)
        print(gene)
        print(muts[list(muts.keys())[0]])
        print()

    def _mutants_positions(self, seqs):
        binary, weights, keys = Convertor(self.setuper).prepare_aligned_msa_for_Vae(seqs)
        data, _ = self.handler.propagate_through_VAE(binary, weights, keys)
        return data

    def create_fasta_file(self):
        with open(self.pickle + "/training_alignment.pkl", 'rb') as file_handle:
            seq_dict = pickle.load(file_handle)
        ## Now transform sequences back to fasta
        with open(self.pickle + "/training_alignment.fasta", 'w') as file_handle:
            for seq_name, seq in seq_dict.items():
                file_handle.write(">" + "".join(seq_name) + "\n" + "".join(seq) + "\n")
        print('Fasta file generate to ', self.pickle + '/training_alignment.fasta')

    def store_ancestors_in_fasta(self, names, file_name='generated_ancestors.fasta'):
        with open(self.pickle + "/" + file_name, 'w') as file_handle:
            for i, seq in enumerate(self.anc_seqs):
                file_handle.write(">" + names[i] + "\n" + "".join(seq) + "\n")
        print('Fasta file generate to ', self.pickle + '/generated_ancestors.fasta')

    def _store_in_fasta_csv(self, to_store, probs, to_file, coords):
        names = list(to_store.keys())
        vals = list(to_store.values())
        self.anc_seqs = vals
        self.store_ancestors_in_fasta(names=names, file_name=to_file)
        # Store in csv file
        with open(self.setuper.high_fld + '{0}_probabilities_ancs.csv'.format(to_file.split('.')[0]), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Number", "Ancestor", "Sequences", "Probability of observation", "Coordinate x", "Coordinate y"])
            for i, (name, seq, prob, c) in enumerate(zip(names, vals, probs, coords)):
                writer.writerow([i, name, seq, prob, c[0], c[1]])

    def _get_ancestor(self, mutants):
        mutants_pos = self._mutants_positions(mutants)
        seqs = list(mutants.values())
        min_dist = float("inf")
        ancestor = []
        anc_pos = []
        for i, p in enumerate(mutants_pos):
            dis = sqrt(p[0]**2 + p[1]**2)
            if dis < min_dist:
                min_dist = dis
                ancestor = seqs[i]
                anc_pos = p

        return min_dist, ancestor, anc_pos, mutants_pos

    def get_straight_ancestors(self, cnt_of_anc=100):
        '''Method does that the line between a position of reference sequence
         and the center (0,0) is divided to cnt_of_anc and the borders of the
         intervals are denoted as ancestors. It can be used as a validation
         metric that int the randomly initialized weights od encoder has no
         effect on latent space if the radiuses of differently init models
         are almost same.
         '''
        lat_dim = self.setuper.dimensionality
        ref_pos = self._mutants_positions(self.cur)
        if lat_dim == 3:
            (x_s, y_s, z_s) = ref_pos[0]
        else:
            (x_s, y_s) = ref_pos[0]
            z_s = 0
        cnt_of_anc += 1
        x_d = (x_s / cnt_of_anc)
        y_d = (y_s / cnt_of_anc)
        z_d = (z_s / cnt_of_anc)
        i = 1
        to_highlight = [(x_s, y_s, z_s)] if lat_dim == 3 else [(x_s, y_s)]
        while i <= cnt_of_anc:
            cur_x = x_s - (x_d * i)
            cur_y = y_s - (y_d * i)
            cur_z = z_s - (z_d * i)
            if lat_dim == 3:
                to_highlight.append((cur_x, cur_y, cur_z))
            else:
                to_highlight.append((cur_x, cur_y))
            i += 1
        ancestors_to_store = self.handler.decode_sequences_VAE(to_highlight, self.cur_name)
        probs = ProbabilityMaker(None, None, self.setuper, generate_negative=False).measure_seq_probability(ancestors_to_store)
        self._store_in_fasta_csv(ancestors_to_store, to_file='straight_ancestors.fasta', probs=probs, coords=to_highlight)
        return list(ancestors_to_store.keys()), to_highlight, probs

    def dynamic_system(self):
        """Ancestor reconstruction is made not straightly
         but by dynamic system described by formula:
            x(t+1) = beta(-sgn(x(t)) + alpha_x)
            y(t+1) = beta(-sgn(y(t)) + alpha_y)

            alpha - stands for weighted values of x/y coordinates
                    in the gaussian space around current point
            beta  - the size of step
        """
        CENTER_THRESHOLD = 0.01
        BETA = self.setuper.dyn_beta

        ancestors = []
        # Load the trained latent space representation
        pickle_name = self.pickle + "/latent_space.pkl"
        if path.isfile(pickle_name):
            print("Mutagenesis message: Latent space file already exists in {0}. \n"
                  "                     Loading that file for dynamic system analyse ...".format(pickle_name))
            with open(pickle_name, 'rb') as file_handle:
                latent_space = pickle.load(file_handle)
        else:
            print('Mutagenesis message: Latent space file does not exist. Cannot do dynamic system analysis')
            return ancestors
        if self.setuper.dimensionality != 2:
            print('Mutagenesis message: Latent space dimensionality is bigger than 2.  Unsupported option for dynamic system analysis')
            return ancestors
        # sort Mus by individual coordinates
        mus = latent_space['mu']
        mus_x_sort = sorted(mus, key=lambda i: i[0])

        # Get starting point for dynamic system
        mean = self._mutants_positions(self.cur)[0]
        cov = [[1, 0], [0, 1]]

        weightor = norm(mean=mean, cov=cov)
        dist_to_center = sqrt(mean[0]**2 + mean[1]**2)
        iterations = 1
        ancestors.append(mean.copy())
        while dist_to_center < CENTER_THRESHOLD and iterations < 4:
            # select for weighting points from reasonable surroundings
            start, end = next((x for x, val in enumerate(mus_x_sort) if val[0] > mean[0]-1), default=0), \
                             next((x for x, val in enumerate(mus_x_sort) if val[0] > mean[0]+1), default=(len(mus_x_sort)))
            # Now sort by second coordinate and select appropriated slice
            selected = sorted(mus_x_sort[start: end, :], key=lambda i: i[1])
            start, end = next((x for x, val in enumerate(selected) if val[1] > mean[1] - 1), default=0), \
                         next((x for x, val in enumerate(selected) if val[1] > mean[1] + 1),default=(len(mus_x_sort)))
            # Now selected array obtains just points in the 1 x 1 square around mean
            selected = selected[:, start: end]
            # Weighted coordinates influence
            alpha_x, alpha_y = 0, 0
            for x, y in selected:
                p_prob = weightor.pdf([x, y])
                alpha_x += x * p_prob
                alpha_y += y * p_prob
            alpha_x, alpha_y = alpha_x / len(selected), alpha_y / len(selected)
            # apply formulas for dynamic system x(t+1) = beta(-sgn(x(t)) + alpha_x)
            mean[0] = BETA * (-np.sign(mean[0]) + alpha_x)
            mean[1] = BETA * (-np.sign(mean[1]) + alpha_y)
            # update probabilistic density function according actual mean
            weightor = norm(mean=mean, cov=cov)
            dist_to_center = sqrt(mean[0] ** 2 + mean[1] ** 2)
            iterations += 1
            ancestors.append(mean.copy())
        ancestors_to_store = self.handler.decode_sequences_VAE(ancestors, self.cur_name)
        probs = ProbabilityMaker(None, None, self.setuper, generate_negative=False).measure_seq_probability(ancestors_to_store)
        self._store_in_fasta_csv(ancestors_to_store, to_file='dynamic_system.fasta', probs=probs,coords=ancestors)
        return ancestors, probs

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    dow = Downloader(tar_dir)
    mut = MutagenesisGenerator(setuper=tar_dir, num_point_mut=tar_dir.mut_points)
    # ancs, names, muts = mut.reconstruct_ancestors(samples=tar_dir.mutant_samples)
    h = Highlighter(tar_dir)
    # h.highlight_mutants(ancs, names, muts, file_name='mutans_{}_{}'.format(tar_dir.mut_points, tar_dir.mutant_samples))
    # if tar_dir.focus:
    #     h.highlight_mutants(ancs, names, muts, file_name='mutans_focus_{}_{}'.format(tar_dir.mut_points, tar_dir.mutant_samples), focus=tar_dir.focus)
    # mut.store_ancestors_in_fasta(names)
    names, ancestors, probs = mut.get_straight_ancestors()
    h.highlight_mutants(ancs=ancestors, names=names, mutants=[], file_name='straight_ancestors_no_focus', focus=False)
    h = Highlighter(tar_dir)
    h.plot_probabilities(probs, ancestors)
    ancestors, probs = mut.dynamic_system()
    h.plot_probabilities(probs, ancestors, dynamic=True)