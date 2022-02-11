__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/18 09:33:12"

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from Bio import pairwise2
from Bio.Align.Applications import ClustalOmegaCommandline

from download_MSA import Downloader
from msa_preparation import MSA
from parser_handler import CmdHandler
from sequence_transformer import Transformer
from supportscripts.animator import GifMaker
from VAE_accessor import VAEAccessor


class Highlighter:
    def __init__(self, setuper):
        self.handler = VAEAccessor(setuper, setuper.get_model_to_load())
        self.mu, self.sigma, self.latent_keys = self.handler.latent_space(check_exists=False)
        self.out_dir = setuper.high_fld + '/'
        self.name = "class_highlight.png"
        self.setuper = setuper
        self.plt = self._init_plot()
        self.transformer = Transformer(setuper)

    def _init_plot(self):
        if self.setuper.dimensionality == 3:
            return self._highlight_3D(name='', high_data=self.mu)
        self.fig, ax = plt.subplots()
        ax.plot(self.mu[:, 0], self.mu[:, 1], '.', alpha=0.1, markersize=3, label='training')
        ax.set_xlim([-7, 7])
        ax.set_ylim([-7, 7])
        ax.set_xlabel("$Z_1$")
        ax.set_ylabel("$Z_2$")
        return ax

    def _highlight(self, name, high_data, one_by_one=False, wait=False, no_init=False, color='red',
                   file_name='ancestors', focus=False):
        plt = self.plt if no_init else self._init_plot()
        if self.setuper.dimensionality == 3:
            self._highlight_3D(name, high_data)
            return
        alpha = 0.2
        if len(high_data) < len(self.mu) * 0.1:
            alpha = 1  ## When low number of points should be highlighted make them brighter
        if one_by_one:
            for name_idx, data in enumerate(high_data):
                plt.plot(data[0], data[1], '.', color='black', alpha=1, markersize=3,
                         label=name[name_idx] + '({})'.format(name_idx))
                plt.annotate(str(name_idx), (data[0], data[1]))
            name = file_name
        else:
            plt.plot(high_data[:, 0], high_data[:, 1], '.', color=color, alpha=alpha, markersize=3, label=name)
        if not wait:
            # Nothing will be appended to plot so generate it
            # Put a legend to the right of the current axis
            # plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
            if focus:
                # now later you get a new subplot; change the geometry of the existing
                x1 = high_data[0][0]
                x2 = high_data[-1][0]
                y1 = high_data[0][1]
                y2 = high_data[-1][1]
                x1, x2 = (x1, x2) if x1 < x2 else (x2, x1)
                y1, y2 = (y1, y2) if y1 < y2 else (y2, y1)
                plt.set_xlim([x1 - 0.5, x2 + 0.5])
                plt.set_ylim([y1 - 0.5, y2 + 0.5])
            else:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.tight_layout()
            # plt.title(label='Filtered Dataset, Weight = {}'.format(self.setuper.decay))
            save_path = self.out_dir + name.replace('/', '-') + '{}_'.format(self.setuper.model_name) + self.name
            print("Class highlighter saving graph to", save_path)
            self.fig.savefig(save_path, bbox_inches='tight')

    def highlight_mutants(self, ancs, names, mutants, mut_names=None, file_name='mutants', focus=False):
        colors = ['green', 'red', 'blue', 'black', 'magenta', 'chocolate', 'tomato', 'orangered', 'sienna']
        self.plt = self._init_plot()
        # Check if names at mutants are given, otherwise init them
        if mut_names is None:
            mut_names = ['' for _ in range(len(mutants))]
        for i, m in enumerate(mutants):
            self._highlight(name=mut_names[i], high_data=m, wait=True, no_init=True, color=colors[i % len(colors)])
        self._highlight(name=names, high_data=ancs, no_init=True, file_name=file_name, one_by_one=True, focus=focus)

    def highlight_line_deviation(self, query_pos, points, mean, maxDev, loss, mu, file_name='robustness_plot'):
        """ Higlight deviation against line for robustness purposes """
        # Setup latent space by this model robust model
        self.fig, ax = plt.subplots()
        ax.plot(mu[:, 0], mu[:, 1], '.', alpha=0.1, markersize=3, label='full')
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
        ax.set_xlabel("$Z_1$")
        ax.set_ylabel("$Z_2$")
        # Draw line for this model from center to query
        x = [0, query_pos[0][0]]
        y = [0, query_pos[0][1]]
        ax.plot(x, y, color='orange')
        ax.set_title(r'Deviation stats mean={0:.2f}, max={1:.2f}, loss={2:.2f}'.format(mean, maxDev, loss))
        self._highlight(name=file_name, high_data=points, no_init=True, color='red')

    def highlight_file(self, file_name, wait=False):
        msa = MSA.load_msa(file=file_name)
        name = (file_name.split("/")[-1]).split(".")[0]
        names = list(msa.keys())
        if self.setuper.align:
            msa = AncestorsHandler(setuper=self.setuper).align_to_ref(msa=msa)
            binary, weights, keys = self.transformer.sequence_dict_to_binary(msa)
            data, _ = self.handler.propagate_through_VAE(binary, weights, keys)
            self._highlight(name=names, high_data=data, one_by_one=True, wait=wait, no_init=True)
        else:
            data = self._name_match(names)
            self._highlight(name=name, high_data=data)

    def highlight_name(self, name):
        data = self._name_match([name])
        self._highlight(name=name, high_data=data, no_init=True)

    def plot_probabilities(self, probs, ancestors, dynamic=False, file_notion=''):
        '''This method plots probabilities of sequences into three plots.'''
        ratios = [9, 3, 5] if self.setuper.align else [6, 1]
        cnt_plot = 3 if self.setuper.align else 2
        fig, ax = plt.subplots(cnt_plot, 1, gridspec_kw={'height_ratios': ratios})
        fig.tight_layout()
        ax[0].plot(self.mu[:, 0], self.mu[:, 1], '.', alpha=0.1, markersize=3, label='full')
        if self.setuper.align:
            msa = MSA.load_msa(file=self.setuper.highlight_files)
            msa = AncestorsHandler(setuper=self.setuper).align_to_ref(msa=msa)
            binary, weights, keys = self.transformer.sequence_dict_to_binary(msa)
            data, _ = self.handler.propagate_through_VAE(binary, weights, keys)
            ## Plot data into both graphs and focus on the area
            ax[2].plot(self.mu[:, 0], self.mu[:, 1], '.', alpha=0.1, markersize=3)
            for i, d in enumerate(data):
                ax[0].plot(d[0], d[1], '.', alpha=1, markersize=4, label='{0} {1}'.format(i + 1, keys[i]), color='red')
                ax[2].plot(d[0], d[1], '.', alpha=1, markersize=4, label='{0} {1}'.format(i + 1, keys[i]), color='red')
            for i, k in enumerate(keys):
                ax[0].annotate(i + 1, data[i, :2])
                ax[2].annotate(i + 1, data[i, :2])
            x1 = min(data[:, 0])
            x2 = max(data[:, 0])
            y1 = min(data[:, 1])
            y2 = max(data[:, 1])
            x1, x2 = (x1, x2) if x1 < x2 else (x2, x1)
            y1, y2 = (y1, y2) if y1 < y2 else (y2, y1)
            # Plot ancestors
            ax[2].plot([a[0] for i, a in enumerate(ancestors) if i % 10 == 0],
                       [a[1] for i, a in enumerate(ancestors) if i % 10 == 0], '.')
            for i, a in enumerate(ancestors):
                if i % 50 == 0:
                    ax[2].annotate(i, a[:2])
            ax[2].set_xlim([x1 - 0.7, x2 + 0.7])
            ax[2].set_ylim([y1 - 0.7, y2 + 0.7])
            ax[2].legend(title='Babkova seqs', bbox_to_anchor=(1.05, 1), loc='upper left')
            # Keep ratio
            ax[2].set(adjustable='box', aspect='equal')
        if dynamic:
            ax[0].plot([a[0] for a in ancestors], [a[1] for a in ancestors], '-o', markersize=1)
        else:
            ax[0].plot([a[0] for i, a in enumerate(ancestors) if i % 10 == 0],
                       [a[1] for i, a in enumerate(ancestors) if i % 10 == 0], '.')
        # for i in range(len(ancestors)):
        #     if i % 10 == 0:
        #         ax[0].annotate(i, (ancestors[i][0], ancestors[i][1]))
        ax[1].plot(list(range(len(probs))), probs, 'bo', list(range(len(probs))), probs, 'k')
        ax[1].set_xlabel("$Sequence number$")
        ax[1].set_ylabel("$Probability$")

        # Keep ratio
        ax[0].set(adjustable='box', aspect='equal')
        # plt.show(
        save_path = self.out_dir + '{0}probability_graph{1}{2}.png'.format('dynamic_' if dynamic
                                                                           else 'aligned_',
                                                                           file_notion,
                                                                           self.setuper.model_name)
        print("Class highlighter saving probability plot to", save_path)
        fig.savefig(save_path, bbox_inches='tight')

    def plot_straight_probs_against_ancestors(self, straight, ancs_probs, ancs_names):
        ''' Method is creating plot as bot line with horizontal lines of anc to include
            probabilities of ancestors given by file. '''
        fig, ax = plt.subplots(1, 1)
        ax.plot(list(range(len(straight))), straight, 'bo', list(range(len(straight))), straight, 'k')
        ax.set_xlabel("$Sequence number$")
        ax.set_ylabel("$Probability$")

        colors = ['green', 'red', 'salmon', 'coral', 'chocolate', 'orangered', 'sienna']
        probs = [(i, j) for i, j in zip(ancs_probs, [x for x in range(1, len(ancs_names) + 1)])]
        sort_probs = sorted(probs, key=lambda x: x[0])

        i = 0
        for anc, n in sort_probs:
            ax.hlines(y=anc, xmin=0, xmax=len(straight), linewidth=1, color=colors[i])
            if i % 2 == 0:
                ax.text(len(straight) + 2, anc, n, ha='left', va='center')
            else:
                ax.text(-2, anc, n, ha='right', va='center')
            i += 1

        save_path = self.out_dir + 'Sebestova_probs_{}.png'.format(self.setuper.model_name)
        print("Class highlighter saving aligned given ancestral sequences probability plot to", save_path)
        fig.savefig(save_path, bbox_inches='tight')

    def _name_match(self, names):
        ## Key to representation of index
        key2idx = {}
        key2idx_reducted = {}
        for i in range(len(self.latent_keys)):
            key2idx[self.latent_keys[i]] = i
            key2idx_reducted[self.latent_keys[i].split('/')[0]] = i
        reducted_names = [name.split('/')[0] for name in
                          names]  ## Remove number after gene name PIP_BREDN/{233-455} <- remove this
        idx = []
        succ = 0
        fail = 0
        for n in names:
            cur_ind = succ + fail
            try:
                idx.append(int(key2idx[n]))
                succ += 1
            except KeyError as e:
                try:
                    idx.append(int(key2idx_reducted[reducted_names[cur_ind]]))
                    if self.setuper.stats:
                        print("Reduce name match")
                    succ += 1
                except KeyError as e:
                    fail += 1  # That seq is missing even in original seq set. Something terrifying is happening here.
        if self.setuper.stats:
            print("=" * 60)
            print("Printing match stats")
            print(" Success: ", succ, " Fails: ", fail)
        return self.mu[idx, :]

    def _highlight_3D(self, name, high_data, color='blue'):
        if name == '':
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel("$Z_1$")
            self.ax.set_ylabel("$Z_2$")
            self.ax.set_zlabel("$Z_3$")
            self.ax.set_xlim(-6, 6)
            self.ax.set_ylim(-6, 6)
            self.ax.set_zlim(-6, 6)
            self.ax.scatter(high_data[:, 0], high_data[:, 1], high_data[:, 2], color='blue', alpha=0.1)
            return self.ax
        self.ax.scatter(high_data[:, 0], high_data[:, 1], high_data[:, 2], color='red')
        gif_name = self.name.replace('.png', '.gif')
        save_path = self.out_dir + name.replace('/', '-') + '_3D_{}'.format(self.setuper.model_name) + gif_name
        GifMaker(self.fig, self.ax, save_path)
        save_path = self.out_dir + name.replace('/', '-') + '_3D_{}'.format(self.setuper.model_name) + self.name
        print("Class highlighter saving 3D graph to", save_path)
        self.fig.savefig(save_path)

    def plot_instrict_dimensions(self, mus):
        '''Plot histograms for individual dimensions to check collapsed one'''
        dim = mus.shape[-1]
        fig, axs = plt.subplots((dim // 6) + 1, 3)
        bins = np.linspace(-6, 6, 30)

        save_path = self.out_dir + "Instrict_dim_{}_".format(self.setuper.model_name)
        for d in range(0, dim, 2):
            axs[(d // 6), (d // 2) % 3].hist(mus[:, d], bins, alpha=0.5, label='dim_{}'.format(d))
            if d + 1 != dim:
                axs[(d // 6), (d // 2) % 3].hist(mus[:, d + 1], bins, alpha=0.5, label='dim_{}'.format(d + 1))
            axs[(d // 6), (d // 2) % 3].legend(loc='upper right')
        save_to = save_path + 'combined.png'
        print("Class highlighter saving graph to", save_to)
        fig.savefig(save_to)


class AncestorsHandler:
    def __init__(self, setuper):
        self.setuper = setuper
        self.pickle = setuper.pickles_fld

    def align_to_ref(self, msa: dict = None):
        """
            Align sequences to original MSA and then apply
            the same position removal as in MSA processing.
            In the case of msa_align False do iterative alignment
            to only query sequence. This may not be optimal solution
        """
        aligned = {}
        # Do iterative alignment only to query (not optimal solution)
        with open(self.pickle + "/reference_seq.pkl", 'rb') as file_handle:
            ref = pickle.load(file_handle)
            ref_name = list(ref.keys())[0]
            ref_seq = "".join(ref[ref_name])
            aligned, ref_len = {}, len(ref_seq)
            i = 0
            for k in msa.keys():
                i += 1
                seq = msa[k]
                alignments = pairwise2.align.globalms(ref_seq, seq, 3, 1, -7, -1)
                best_align = alignments[0][1]
                if len(seq) > ref_len:
                    # length of sequence is bigger than ref query, cut sequence on reference query gap positions
                    print(' AncestorHandler message: Len of seq is {0}, length of reference is {1}.\n '
                          '                          Sequence amino position'
                          '  at reference gaps will be removed'.format(len(seq), ref_len))
                    tmp = ''
                    len_dif = len(best_align) - ref_len
                    aligned_query = alignments[0][0]
                    idx_to_remove = []
                    # Remove gap positions when occur in both query and aligned sequence
                    for i in range(len(aligned_query)):
                        if aligned_query[i] == '-' and best_align[i] == '-' and len_dif > 0:
                            len_dif -= 1
                            idx_to_remove.append(i)
                    if len_dif > 0:
                        # Remove positions where aligned query has gaps
                        for i in range(len(aligned_query)):
                            if aligned_query[i] == '-' and best_align[i] == '-' and len_dif > 0:
                                len_dif -= 1
                                idx_to_remove.append(i)
                            len_dif -= 1
                    aligned[k] = tmp.join([best_align[i] for i in range(len(best_align)) if i not in idx_to_remove])
                else:
                    # try 3 iteration to fit ref query
                    open_gap_pen, gap_pen = -7, -1
                    while len(best_align) > ref_len:
                        open_gap_pen, gap_pen = open_gap_pen - 1, gap_pen - 1
                        alignments = pairwise2.align.globalms(ref_seq, seq, 3, 1, open_gap_pen, gap_pen)
                        best_align = alignments[0][1]
                    aligned[k] = alignments[0][1]
            if self.setuper.stats:
                print(k, ':', alignments[0][2], len(aligned[k]))
        return aligned

    def align_to_original_msa(self, msa_file):
        """
        Align msa passed via fasta file to original MSA using ClustalOmega
        Align sequences to the profile of original MSA and filter by same protocol
        """
        aligned = {}

        import multiprocessing
        cores_count = min(multiprocessing.cpu_count(), 8)

        msa = MSA.load_msa(msa_file)

        ancestral_names = list(msa.keys())
        # check if alignment exists
        outfile = self.pickle + "/{}_aligned_to_MSA.fasta".format(msa_file.split(".")[0])
        if os.path.exists(outfile) and os.path.getsize(outfile) > 0:
            print(' Analyzer message : Alignment file exists in {}. Using that file.'.format(outfile))
        else:
            # Create profile from sequences to be aligned
            profile = self.pickle + "/ancestors_align.fasta"
            if os.path.exists(profile) and os.path.getsize(profile) > 0:
                print(' Analyzer message : Alignment of ancestors exists in {}. Using that file.'.format(profile))
            else:
                clustalomega_cline = ClustalOmegaCommandline(cmd=self.setuper.clustalo_path,
                                                             infile=msa_file,
                                                             outfile=profile,
                                                             threads=cores_count,
                                                             verbose=True, auto=True)  # , dealign=True)
                print(" AncestorHandler message : Aligning ancestors ...\n"
                      "                           Running {}".format(clustalomega_cline))
                stdout, stderr = clustalomega_cline()

            clustalomega_cline = ClustalOmegaCommandline(cmd=self.setuper.clustalo_path,
                                                         profile1=self.setuper.in_file,
                                                         profile2=profile,
                                                         outfile=outfile,
                                                         threads=cores_count,
                                                         verbose=True, auto=True)  # , isprofile=True)
            print(" AncestorHandler message : Running {}".format(clustalomega_cline))
            stdout, stderr = clustalomega_cline()
            print(stdout)
        with open(self.pickle + "/seq_pos_idx.pkl", 'rb') as file_handle:
            pos_idx = pickle.load(file_handle)
        # Find sequences from file and process them in the same way as during MSA processing
        alignment = MSA.load_msa(outfile)
        for name in ancestral_names:
            aligned[name] = [item for i, item in enumerate(alignment[name]) if i in pos_idx]
        return aligned


if __name__ == '__main__':
    tar_dir = CmdHandler()
    down_MSA = Downloader(tar_dir)
    ## Create latent space
    mus, _, _ = VAEAccessor(setuper=tar_dir, model_name=tar_dir.get_model_to_load()).latent_space()
    ## Highlight
    highlighter = Highlighter(tar_dir)
    if tar_dir.highlight_files is not None:
        files = tar_dir.highlight_files.split()
        wait_high = True if len(files) == 1 and tar_dir.highlight_seqs is not None else False
        for f in files:
            highlighter.highlight_file(file_name=f, wait=wait_high)
    if tar_dir.highlight_seqs is not None:
        names = tar_dir.highlight_seqs.split()
        for n in names:
            highlighter.highlight_name(name=n)
    if tar_dir.highlight_instricts:
        highlighter.plot_instrict_dimensions(mus)
