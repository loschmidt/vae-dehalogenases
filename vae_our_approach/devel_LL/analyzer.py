__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/18 09:33:12"

from pipeline import StructChecker
from torch.utils.data import Dataset, DataLoader
from VAE_model import MSA_Dataset, VAE
from msa_prepar import MSA

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import os.path as path


class Highlighter:
    def __init__(self, setuper):
        self.mu, self.sigma, self.latent_keys = VAEHandler(setuper).latent_space(check_exists=True)
        self.out_dir = setuper.high_fld + '/'
        self.name = "class_highlight.png"
        self.setuper = setuper

    def _init_plot(self):
        plt.figure(0)
        plt.clf()
        plt.plot(self.mu[:, 0], self.mu[:, 1], '.', alpha=0.1, markersize=3, label='full')
        plt.xlim((-6, 6))
        plt.ylim((-6, 6))
        plt.xlabel("$Z_1$")
        plt.ylabel("$Z_2$")
        plt.legend(loc="upper left")
        plt.tight_layout()
        return plt

    def _highlight(self, name, high_data):
        plt = self._init_plot()
        alpha = 0.2
        if len(high_data) < len(self.mu) * 0.1:
            alpha = 1 ## When low number of points should be highlighted make them brighter
        plt.plot(high_data[:, 0], high_data[:, 1], '.',color='red', alpha=alpha, markersize=3, label=name)
        save_path = self.out_dir + name.replace('/','-') + '_' + self.name
        print("Class highlighter saving graph to", save_path)
        plt.savefig(save_path)

    def highlight_file(self, file_name):
        msa = MSA(setuper=self.setuper, processMSA=False).load_msa(file=file_name)
        name = (file_name.split("/")[-1]).split(".")[0]
        names = list(msa.keys())
        data = self._name_match(names)
        self._highlight(name=name, high_data=data)

    def highlight_name(self, name):
        data = self._name_match([name])
        self._highlight(name=name, high_data=data)

    def _name_match(self, names):
        ## Key to representation of index
        key2idx = {}
        key2idx_reducted = {}
        for i in range(len(self.latent_keys)):
            key2idx[self.latent_keys[i]] = i
            key2idx_reducted[self.latent_keys[i].split('/')[0]] = i
        reducted_names = [name.split('/')[0] for name in names] ## Remove number after gene name PIP_BREDN/{233-455} <- remove this
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

class VAEHandler:
    def __init__(self, setuper):
        self.setuper = setuper
        self.pickle = setuper.pickles_fld
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
        vae = VAE(num_res_type, 2, len_protein * num_res_type, [100])
        ## move the VAE onto a GPU
        if self.use_cuda:
            vae.cuda()
        vae.load_state_dict(torch.load(self.setuper.VAE_model_dir + "/vae_0.01_fold_0.model"))
        return vae, msa_binary, num_seq

    def _load_pickles(self):
        with open(self.pickle + "/keys_list.pkl", 'rb') as file_handle:
            msa_keys = pickle.load(file_handle)
        with open(self.pickle + "/seq_weight.pkl", 'rb') as file_handle:
            seq_weight = pickle.load(file_handle)
        return seq_weight, msa_keys

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
        return mu, sigma, key

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    ## Create latent space
    VAEHandler(setuper=tar_dir).latent_space()
    ## Highlight
    if tar_dir.highlight_files is not None:
        highlighter = Highlighter(tar_dir)
        files = tar_dir.highlight_files.split()
        for f in files:
            highlighter.highlight_file(file_name=f)
    if tar_dir.highlight_seqs is not None:
        highlighter = Highlighter(tar_dir)
        names = tar_dir.highlight_seqs.split()
        for n in names:
            highlighter.highlight_name(name=n)
