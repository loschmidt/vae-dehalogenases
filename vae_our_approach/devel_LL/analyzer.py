__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/18 09:33:12"

from Bio import pairwise2
from Bio.Align.Applications import ClustalOmegaCommandline
from download_MSA import Downloader
from pipeline import StructChecker
from torch.utils.data import Dataset, DataLoader
from VAE_model import MSA_Dataset, VAE
from msa_prepar import MSA
from msa_filter_scorer import MSAFilterCutOff as Convertor
from supportscripts.animator import GifMaker
from torch import tensor

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import os.path as path


class Highlighter:
    def __init__(self, setuper):
        self.handler = VAEHandler(setuper)
        self.mu, self.sigma, self.latent_keys = self.handler.latent_space(check_exists=True)
        self.out_dir = setuper.high_fld + '/'
        self.name = "class_highlight.png"
        self.setuper = setuper
        self.plt = self._init_plot()

    def _init_plot(self):
        if self.setuper.dimensionality == 3:
            return self._highlight_3D(name='', high_data=self.mu)
        self.fig, ax = plt.subplots()
        ax.plot(self.mu[:, 0], self.mu[:, 1], '.', alpha=0.1, markersize=3, label='full')
        ax.set_xlim([-6, 6])
        ax.set_ylim([-6, 6])
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
            #plt.title(label='Filtered Dataset, Weight = {}'.format(self.setuper.decay))
            save_path = self.out_dir + name.replace('/', '-') + '{}_'.format(self.setuper.model_name) + self.name
            print("Class highlighter saving graph to", save_path)
            self.fig.savefig(save_path, bbox_inches='tight')

    def highlight_mutants(self, ancs, names, mutants, mut_names=None, file_name='mutants', focus=False):
        colors = ['green', 'red', 'salmon', 'blue', 'coral', 'chocolate', 'tomato', 'orangered', 'sienna']
        self.plt = self._init_plot()
        # Check if names at mutants are given, otherwise init them
        if mut_names is None:
            mut_names = ['' for _ in range(len(mutants))]
        for i, m in enumerate(mutants):
            self._highlight(name=mut_names[i], high_data=m, wait=True, no_init=True, color=colors[i % len(colors)])
        self._highlight(name=names, high_data=ancs, no_init=True, file_name=file_name, one_by_one=True, focus=focus)

    def highlight_file(self, file_name, wait=False):
        msa = MSA(setuper=self.setuper, processMSA=False).load_msa(file=file_name)
        name = (file_name.split("/")[-1]).split(".")[0]
        names = list(msa.keys())
        if self.setuper.align:
            msa = AncestorsHandler(setuper=self.setuper, seq_to_align=msa).align_to_ref()
            binary, weights, keys = Convertor(self.setuper).prepare_aligned_msa_for_Vae(msa)
            data, _ = self.handler.propagate_through_VAE(binary, weights, keys)
            self._highlight(name=names, high_data=data, one_by_one=True, wait=wait, no_init=True)
        else:
            data = self._name_match(names)
            self._highlight(name=name, high_data=data)

    def highlight_name(self, name):
        data = self._name_match([name])
        self._highlight(name=name, high_data=data, no_init=True)

    def plot_probabilities(self, probs, ancestors, dynamic=False, file_notion=''):
        ratios = [9, 3, 5] if self.setuper.align else [6, 1]
        cnt_plot = 3 if self.setuper.align else 2
        fig, ax = plt.subplots(cnt_plot, 1, gridspec_kw={'height_ratios': ratios})
        ax[0].plot(self.mu[:, 0], self.mu[:, 1], '.', alpha=0.1, markersize=3, label='full')
        if self.setuper.align:
            msa = MSA(setuper=self.setuper, processMSA=False).load_msa(file=self.setuper.highlight_files)
            msa = AncestorsHandler(setuper=self.setuper, seq_to_align=msa).align_to_ref()
            binary, weights, keys = Convertor(self.setuper).prepare_aligned_msa_for_Vae(msa)
            data, _ = self.handler.propagate_through_VAE(binary, weights, keys)
            ## Plot data into both graphs and focus on the area
            ax[2].plot(self.mu[:, 0], self.mu[:, 1], '.', alpha=0.1, markersize=3)
            for i, d in enumerate(data):
                ax[0].plot(d[0], d[1], '.', alpha=1, markersize=4, label='{0} {1}'.format(i + 1, keys[i]), color='red')
                ax[2].plot(d[0], d[1], '.', alpha=1, markersize=4, label='{0} {1}'.format(i + 1, keys[i]), color='red')
            for i, k in enumerate(keys):
                ax[0].annotate(i + 1, data[i, :2])
                ax[2].annotate(i + 1, data[i, :2])
            x1 = min(data[:,0])
            x2 = max(data[:,0])
            y1 = min(data[:,1])
            y2 = max(data[:,1])
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
        from mpl_toolkits.mplot3d import Axes3D
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
        fig, axs = plt.subplots((dim//6)+1, 3)
        bins = np.linspace(-6, 6, 30)

        save_path = self.out_dir + "Instrict_dim_{}_".format(self.setuper.model_name)
        for d in range(0, dim, 2):
            axs[(d // 6), (d // 2) % 3].hist(mus[:, d], bins, alpha=0.5, label='dim_{}'.format(d))
            if d+1 != dim:
                axs[(d // 6), (d // 2) % 3].hist(mus[:, d+1], bins, alpha=0.5, label='dim_{}'.format(d+1))
            axs[(d // 6), (d // 2) % 3].legend(loc='upper right')
        save_to = save_path + 'combined.png'
        print("Class highlighter saving graph to", save_to)
        fig.savefig(save_to)

class VAEHandler:
    def __init__(self, setuper):
        self.setuper = setuper
        self.pickle = setuper.pickles_fld
        self.vae = None
        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

    def _prepare_model(self):
        ## prepare model to mapping from highlighting files
        with open(self.pickle + "/seq_msa_binary.pkl", 'rb') as file_handle:
            msa_original_binary = pickle.load(file_handle)
        num_seq = msa_original_binary.shape[0]
        len_protein = msa_original_binary.shape[1]
        num_res_type = msa_original_binary.shape[2]

        msa_binary = msa_original_binary.reshape((num_seq, -1))
        msa_binary = msa_binary.astype(np.float32)

        ## build a VAE model
        vae = VAE(num_res_type, self.setuper.dimensionality, len_protein * num_res_type, self.setuper.layers)
        vae.load_state_dict(torch.load(self.setuper.VAE_model_dir + "/vae_{}_fold_0_C_{}_D_{}_{}.model"
                                       .format(str(self.setuper.decay), self.setuper.C, self.setuper.dimensionality,
                                               self.setuper.layersString)))
        # vae.load_state_dict(torch.load(self.setuper.VAE_model_dir + "/vae_0.01_fold_0_C_{}_D_{}.model"
        #                               .format(self.setuper.C, self.setuper.dimensionality)))
        # vae.load_state_dict(torch.load(self.setuper.VAE_model_dir + "/vae_0.01_fold_0.model"))
        ## move the VAE onto a GPU
        if self.use_cuda:
            vae.cuda()
        return vae, msa_binary, num_seq

    def _load_pickles(self):
        '''Side effect setups self.vae'''
        with open(self.pickle + "/keys_list.pkl", 'rb') as file_handle:
            msa_keys = pickle.load(file_handle)
        with open(self.pickle + "/seq_weight.pkl", 'rb') as file_handle:
            seq_weight = pickle.load(file_handle)
        return seq_weight.astype(np.float32), msa_keys

    def latent_space(self, check_exists=False):
        pickle_name = self.pickle + "/latent_space.pkl"
        if check_exists and path.isfile(pickle_name):
            print("Latent space file already exists in {0}. \n"
                  "Loading and returning that file...".format(pickle_name))
            with open(pickle_name, 'rb') as file_handle:
                latent_space = pickle.load(file_handle)
            return latent_space['mu'], latent_space['sigma'], latent_space['key']

        msa_weight, msa_keys = self._load_pickles()
        vae, msa_binary, num_seq = self._prepare_model()
        self.vae = vae

        batch_size = num_seq
        train_data = MSA_Dataset(msa_binary, msa_weight, msa_keys)
        train_data_loader = DataLoader(train_data, batch_size=batch_size)

        mu_list = []
        sigma_list = []
        for idx, data in enumerate(train_data_loader):
            msa, weight, key = data
            with torch.no_grad():
                if self.use_cuda:
                    msa = msa.cuda()
                mu, sigma = vae.encoder(msa)
                if self.use_cuda:
                    mu = mu.cpu().data.numpy()
                    sigma = sigma.cpu().data.numpy()
                mu_list.append(mu)
                sigma_list.append(sigma)
        mu = np.vstack(mu_list)
        sigma = np.vstack(sigma_list)

        with open(self.pickle + "/latent_space.pkl", 'wb') as file_handle:
            pickle.dump({'key': key, 'mu': mu, 'sigma': sigma}, file_handle)
        print('The latent space was created....')
        return mu, sigma, key

    def propagate_through_VAE(self, binaries, weights, keys):
        # check if VAE is already ready from latent space method
        vae = self.vae
        if vae is None:
            vae, _, _ = self._prepare_model()

        binaries = binaries.astype(np.float32)
        binaries = binaries.reshape((binaries.shape[0], -1))
        weights = weights.astype(np.float32)
        train_data = MSA_Dataset(binaries, weights, keys)
        train_data_loader = DataLoader(train_data, batch_size=binaries.shape[0])
        mu_list = []
        sigma_list = []
        for idx, data in enumerate(train_data_loader):
            msa, weight, key = data
            with torch.no_grad():
                if self.use_cuda:
                    msa = msa.cuda()
                mu, sigma = vae.encoder(msa)
                if self.use_cuda:
                    mu = mu.cpu().data.numpy()
                    sigma = sigma.cpu().data.numpy()
                mu_list.append(mu)
                sigma_list.append(sigma)
        mu = np.vstack(mu_list)
        sigma = np.vstack(sigma_list)
        return mu, sigma

    def decode_sequences_VAE(self, lat_sp_pos, ref_name):
        # check if VAE is already ready from latent space method
        vae = self.vae
        if vae is None:
            vae, msa_binary, num_seq = self._prepare_model()
        num_seqs = {}
        for i, z in enumerate(lat_sp_pos, 1):
            anc_name = 'ancestor_{}'.format(i) if i > 1 else ref_name
            z = tensor(z)
            if self.use_cuda:
                z = z.cuda()
            num_seqs[anc_name] = vae.decoder_seq(z)
        if len(num_seqs.keys()) == 1:
            # Special case for leangth 1
            num_seqs[anc_name] = num_seqs[anc_name]
        # Convert from numbers to amino acid sequence
        anc_dict = Convertor(self.setuper).back_to_amino(num_seqs)
        return anc_dict

    def decode_for_marginal_prob(self, z, sigma, samples):
        '''Decode binary representation of z. Method optimilized for
         marginal probability computation'''
        vae = self.vae
        if vae is None:
            vae, _, _ = self._prepare_model()
        with torch.no_grad():
            if self.use_cuda:
                z = z.cuda()
                sigma = sigma.cuda()
            # indices already on cpu(not tensor)
            if samples == -1:
                # Decode exact value of z
                ret = vae.decoder_seq(z)
            else:
                ret = vae.decode_samples(z, sigma, samples)
        return ret

    def get_marginal_probability(self, x):
        '''This method returns the exact probability
         as it is obtained by VAE'''
        vae = self.vae
        if vae is None:
            vae, _, _ = self._prepare_model()
        with torch.no_grad():
            if self.use_cuda:
                x = x.cuda()
            # indices already on cpu(not tensor)
            ret = vae.marginal_sequence(x)
        return ret


class AncestorsHandler:
    def __init__(self, setuper, seq_to_align):
        self.sequences = seq_to_align
        self.setuper = setuper
        self.pickle = setuper.pickles_fld

    def align_to_ref(self, msa_align=True):
        '''
            Align sequences to original MSA and then apply
            the same position removal as in MSA processing.
            In the case of msa_align False do iterative alignment
            to only query sequence. This may not be optimal solution
        '''
        aligned = {}
        if not msa_align:
            # Do iterative alignment only to query (not optimal solution)
            with open(self.pickle + "/reference_seq.pkl", 'rb') as file_handle:
                ref = pickle.load(file_handle)
                ref_name = list(ref.keys())[0]
                ref_seq = "".join(ref[ref_name])
                aligned, ref_len = {}, len(ref_seq)
                i = 0
                for k in self.sequences.keys():
                    i += 1
                    seq = self.sequences[k]
                    alignments = pairwise2.align.globalms(ref_seq, seq, 3, 1, -7, -1)
                    best_align = alignments[0][1]
                    if len(seq) > ref_len:
                        # length of sequence is bigger than ref query, cut sequence on reference query gap positions
                        print('AncestorHandler message: Len of seq is {0}, length of reference is {1}.\n '
                            '                         Sequence amino position'
                            ' at reference gaps will be removed'.format(len(seq), ref_len))
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
        else:
            # Need to do one by one from ancestors
            file = self.setuper.highlight_files
            from msa_prepar import MSA
            m = MSA(self.setuper, processMSA=False)
            ancs = m.load_msa(file)
            for k, v in ancs.items():
                # Create tmp file with individual profiles
                profile = "profile_{}.fa".format(k)
                with open(profile, "w") as f:
                    f.write(">" + k + "\n" + v + "\n")
                outfile = self.pickle + "/aligned_{}.aln".format(k)
                clustalomega_cline = ClustalOmegaCommandline(cmd=self.setuper.clustalo_path,
                                                             profile1=self.setuper.in_file,
                                                             profile2=profile,
                                                             outfile=outfile,
                                                             verbose=True, auto=True)
                print("AncestorHandler message : Running {}".format(clustalomega_cline))
                stdout, stderr = clustalomega_cline()
                print(stdout)
        return aligned


if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    down_MSA = Downloader(tar_dir)
    ## Create latent space
    mus, _, _ = VAEHandler(setuper=tar_dir).latent_space()
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
    if tar_dir.highlight_instricts is not None:
        highlighter.plot_instrict_dimensions(mus)
