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
from torch import tensor

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
        with open(self.setuper.high_fld + "/" + file_name, 'w') as file_handle:
            for i, seq in enumerate(self.anc_seqs):
                file_handle.write(">" + names[i] + "\n" + "".join(seq) + "\n")
        print('Fasta file generate to ', self.setuper.high_fld + "/" + file_name)

    def _store_in_fasta_csv(self, to_store, probs, to_file, coords):
        names = list(to_store.keys())
        vals = list(to_store.values())
        self.anc_seqs = vals
        self.store_ancestors_in_fasta(names=names, file_name=to_file)
        # Measuring the identity
        query_name = self.setuper.ref_n
        query_dict = MutagenesisGenerator.binary_to_seq(self.setuper, seq_key=query_name, return_binary=False)
        query_seq = query_dict[query_name]
        identity = lambda x, query: (sum([1 if i==j else 0 for i,j in zip(x,query)])/len(query))*100
        # Store in csv file
        with open(self.setuper.high_fld + '/{0}_probabilities_ancs.csv'.format(to_file.split('.')[0]), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Number", "Ancestor", "Sequences", "Probability of observation", "Coordinate x",
                             "Coordinate y", "Query identity [%]"])
            for i, (name, seq, prob, c) in enumerate(zip(names, vals, probs, coords)):
                seq_str = ''
                query_iden = identity(seq, query_seq)
                writer.writerow([i, name, seq_str.join(seq), prob, c[0], c[1], query_iden])

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
        print("Mutagenesis message : Straight ancestors generating process started")
        ref_pos = self._mutants_positions(self.cur)
        coor_tuple = tuple(ref_pos[0])
        tuple_dim = range(len(ref_pos[0]))
        step_list = [0.0 for _ in tuple_dim]
        for i in tuple_dim:
            step_list[i] = (coor_tuple[i] / cnt_of_anc)
        i = 1
        to_highlight = [coor_tuple]
        while i <= cnt_of_anc:
            cur_coor = [0.0 for _ in tuple_dim]
            for s_i in tuple_dim:
                cur_coor[s_i] = coor_tuple[s_i] - (step_list[s_i] * i)
            to_highlight.append(tuple(cur_coor))
            i += 1
        ancestors_to_store = self.handler.decode_sequences_VAE(to_highlight, self.cur_name)
        probs = ProbabilityMaker(None, None, self.setuper, generate_negative=False).measure_seq_probability(ancestors_to_store)
        file_name = '{}straight_ancestors.fasta'.format(self.setuper.model_name)
        self._store_in_fasta_csv(ancestors_to_store, to_file=file_name, probs=probs, coords=to_highlight)
        return list(ancestors_to_store.keys()), to_highlight, probs

    def dynamic_system(self, beta=None):
        """Ancestor reconstruction is made not straightly
         but by dynamic system described by formula:
            x(t+1) = x(t) + beta(-sgn(x(t)) + alpha_x)
            y(t+1) = y(t) + beta(-sgn(y(t)) + alpha_y)

            alpha - stands for weighted values of x/y coordinates
                    in the gaussian space around current point
            beta  - the size of step
        """
        CENTER_THRESHOLD = 0.01
        BETA = beta if beta is not None else self.setuper.dyn_beta

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
        print('Dynamic process has begun with BETA', BETA)
        # sort Mus by individual coordinates
        mus = latent_space['mu']
        mus_x_sort = sorted(mus, key=lambda i: i[0])

        # Get starting point for dynamic system
        mean = self._mutants_positions(self.cur)[0]
        cov = [[1, 0], [0, 1]]
        # Normal distribution is loacated on 0, 0 coordinates
        weightor = norm(mean=[0,0], cov=cov)
        dist_to_center = sqrt(mean[0]**2 + mean[1]**2)
        iterations = 1
        ancestors.append(mean.copy())
        while dist_to_center > CENTER_THRESHOLD and iterations < 300:
            # select for weighting points from reasonable surroundings
            start, end = next((x for x, val in enumerate(mus_x_sort) if val[0] > mean[0]-1), 0), \
                             next((x for x, val in enumerate(mus_x_sort) if val[0] > mean[0]+1), len(mus_x_sort))
            # Now sort by second coordinate and select appropriated slice
            selected = sorted(mus_x_sort[start: end], key=lambda i: i[1])
            start, end = next((x for x, val in enumerate(selected) if val[1] > mean[1] - 1), 0), \
                         next((x for x, val in enumerate(selected) if val[1] > mean[1] + 1), len(mus_x_sort))
            # Now selected array obtains just points in the 1 x 1 square around mean
            selected = selected[start: end]
            # Weighted coordinates influence
            alpha_x, alpha_y = 0, 0
            for x, y in selected:
                # Transform points to 0, 0 mean normal distribution
                x, y = x - mean[0], y - mean[1]
                p_prob = weightor.pdf([x, y])
                alpha_x += x * p_prob
                alpha_y += y * p_prob
            ## TODO diky normalizaci nemaji jine body vliv a sgn ma vliv solve it 
            #alpha_x, alpha_y = alpha_x / len(selected), alpha_y / len(selected)
            # apply formulas for dynamic system x(t+1) = beta(-sgn(x(t)) + alpha_x)
            mean[0] += BETA * (-np.sign(mean[0]) + alpha_x)
            mean[1] += BETA * (-np.sign(mean[1]) + alpha_y)
            dist_to_center = sqrt(mean[0] ** 2 + mean[1] ** 2)
            iterations += 1
            ancestors.append(mean.copy())
        ancestors_to_store = self.handler.decode_sequences_VAE(ancestors, self.cur_name)
        probs = ProbabilityMaker(None, None, self.setuper, generate_negative=False).measure_seq_probability(ancestors_to_store)
        self._store_in_fasta_csv(ancestors_to_store, to_file='dynamic_system_{}.fasta'.format(str(BETA)), probs=probs,coords=ancestors)
        return ancestors, probs

    def test_generative(self):
        tmp_str = ''
        to_highlight = []

        self.cur[self.cur_name] = [a for a in '---ISAEFPFESKY--VLGSRMHYVDEG---GDP-VLFLHGNPTSSYLWRNIIPHVS--GRCIAPDLIGMGKSDK-PDIDYRFADHARYLDAFIDALGL--ITLVGHDWGSALGFDYAARHPDRVRGIAFMEAILP--P--SW-EFP--ARELFQRFRTP-VGEKMILE-NIFVERVLP--V-VR-PL-TEE-EM-AHYRA-PFPT-PESR-KPL-L-RWP-R-EIPIAG-PADVAEI-VEAYNAWLAA--D-IPKLLFYAEPGVIV-P---V-AWC-AENLPNLEV--DLGPGLHFIQ-EDHPHAIGQA-IADWLRRL-']
        # self.cur[self.cur_name].extend(['K' for _ in range(160)])
        ref_pos = self._mutants_positions(self.cur)
        to_highlight.append(ref_pos)
        print('Ref name', self.cur_name, 'coordinates in the latent space', ref_pos[0])
        print('Reference sequence\n', tmp_str.join(self.cur[self.cur_name]))
        # ancestors_to_store = Convertor(self.setuper).back_to_amino({'first_exam' : idxs[0]})
        ancestors_to_store = self.handler.decode_sequences_VAE(to_highlight, self.cur_name)
        tmp_str = ''
        print('Reference sequence\n', tmp_str.join(ancestors_to_store[list(ancestors_to_store.keys())[0]]))

    @staticmethod
    def binary_to_seq(setuper, seq_key=None, binary=None, return_binary=False):
        '''If seq_keys set the search mode is applied'''
        get_aas = lambda xs: [np.where(aa_bin==1)[0][0] for aa_bin in xs]

        if seq_key is not None:
            print('Searching for given sequence {} .....'.format(seq_key))
            with open(setuper.pickles_fld + "/keys_list.pkl", 'rb') as file_handle:
                keys_list = pickle.load(file_handle)
            with open(setuper.pickles_fld + "/seq_msa_binary.pkl", 'rb') as file_handle:
                msa_original_binary = pickle.load(file_handle)
            key_index = -1
            for i, k in enumerate(keys_list):
                if k == seq_key:
                    key_index = i
                    break
            if key_index == -1:
                print('Key', seq_key, ' not found in list')
                exit(0)
            binary = msa_original_binary[key_index]
        else:
            seq_key = 'default_search_seq'

        nn_aa = get_aas(binary)
        anc_dict = Convertor(setuper).back_to_amino({seq_key : nn_aa})
        return binary if return_binary else anc_dict


if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    dow = Downloader(tar_dir)

    # ref = MutagenesisGenerator.binary_to_seq(tar_dir,seq_key='P27652.1')
    ref_binary = MutagenesisGenerator.binary_to_seq(tar_dir, seq_key='P59336_S14', return_binary=True)
    mut = MutagenesisGenerator(setuper=tar_dir, num_point_mut=tar_dir.mut_points)
    # ancs, names, muts = mut.reconstruct_ancestors(samples=tar_dir.mutant_samples)
    #h = Highlighter(tar_dir)
    # h.highlight_mutants(ancs, names, muts, file_name='mutans_{}_{}'.format(tar_dir.mut_points, tar_dir.mutant_samples))
    # if tar_dir.focus:
    #     h.highlight_mutants(ancs, names, muts, file_name='mutans_focus_{}_{}'.format(tar_dir.mut_points, tar_dir.mutant_samples), focus=tar_dir.focus)
    # mut.store_ancestors_in_fasta(names)
    names, ancestors, probs = mut.get_straight_ancestors()
    #h.highlight_mutants(ancs=ancestors, names=names, mutants=[], file_name='straight_ancestors_no_focus', focus=False)
    h = Highlighter(tar_dir)
    h.plot_probabilities(probs, ancestors)
    # betas = [0.20, 0.15, 0.1, 0.05, 0.08, 0.03]
    # betas = [0.25]
    # for beta in betas:
    #     ancestors, probs = mut.dynamic_system(beta=beta)
    #     h.plot_probabilities(probs, ancestors, dynamic=True, file_notion='_beta_{}'.format(str(beta)))
    # mut.test_generative()
