__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/28 13:49:00"

import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from VAE_model import VAE, MSA_Dataset
from vae_models.cnn_vae import VaeCnn
from vae_models.conditional_vae import CVAE
from msa_handlers.download_MSA import Downloader
from parser_handler import CmdHandler
from project_enums import Helper


class Train:
    def __init__(self, setuper: CmdHandler, benchmark=False):
        self.setuper = setuper
        self.seq_msa_binary, self.seq_weight, self.seq_keys = self.load_pickles()

        self.num_seq = self.seq_msa_binary.shape[0]
        self.len_protein = self.seq_msa_binary.shape[1]
        self.num_res_type = self.seq_msa_binary.shape[2]

        self.K = setuper.K
        self.train_subsets = setuper.train_subsets

        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

        percentage = 20
        self.benchmark = benchmark
        self.benchmark_set = np.zeros((self.num_seq // percentage, self.len_protein, self.num_res_type))
        if benchmark:
            # Take 5 percent from original MSA for further evaluation
            random_idx = np.random.permutation(range(1, self.num_seq))
            benchmark_indices = np.array([])
            for i in range(self.num_seq // percentage):
                self.benchmark_set[i] = self.seq_msa_binary[random_idx[i]]
                benchmark_indices = np.append(benchmark_indices, random_idx[i])
            self.seq_msa_binary = np.delete(self.seq_msa_binary, random_idx[:(self.num_seq // percentage)], axis=0)
            training_weights = np.delete(self.seq_weight, random_idx[:(self.num_seq // percentage)], axis=0)
            training_keys = np.delete(self.seq_keys, random_idx[:(self.num_seq // percentage)], axis=0)
            self.num_seq = self.seq_msa_binary.shape[0]
            self.keys = training_keys
            with open(setuper.pickles_fld + '/positive_control.pkl', 'wb') as file_handle:
                pickle.dump(self.benchmark_set, file_handle)
            with open(setuper.pickles_fld + '/training_set.pkl', 'wb') as file_handle:
                pickle.dump(self.seq_msa_binary, file_handle)
            with open(setuper.pickles_fld + '/training_keys.pkl', 'wb') as file_handle:
                pickle.dump(training_keys, file_handle)
            with open(setuper.pickles_fld + '/training_weights.pkl', 'wb') as file_handle:
                pickle.dump(training_weights, file_handle)
            if setuper.solubility_file:
                with open(setuper.pickles_fld + '/solubilities.pkl', 'rb') as file_handle:
                    solubility = pickle.load(file_handle)
                solubility_positive = solubility[benchmark_indices.astype(int)]
                self.solubility = np.delete(solubility, random_idx[:(self.num_seq // percentage)], axis=0)
                with open(setuper.pickles_fld + '/solubility_positive.pkl', 'wb') as file_handle:
                    pickle.dump(solubility_positive, file_handle)
                with open(setuper.pickles_fld + '/solubility_train_set.pkl', 'wb') as file_handle:
                    pickle.dump(self.solubility, file_handle)
        self.seq_msa_binary = self.seq_msa_binary.reshape((self.num_seq, -1))
        self.seq_msa_binary = self.seq_msa_binary.astype(np.float32)

    def train(self):
        num_seq_subset = self.num_seq // (self.train_subsets + 1)
        idx_subset = []
        random_idx = np.random.permutation(range(self.num_seq))
        for i in range(self.train_subsets):
            idx_subset.append(random_idx[i * num_seq_subset:(i + 1) * num_seq_subset])

        # the following list holds the elbo values on validation data
        elbo_all_list = []

        # Count of validations run is for robustness just one
        K = self.K  #1 if self.setuper.robustness_train else self.K  # train with same subset
        for k in range(K):
            print("Start the {}th fold training".format(k))
            print("-" * 60)

            # build a VAE model with random parameters
            if self.setuper.convolution:
                vae = VaeCnn(self.setuper.dimensionality, self.len_protein)
            elif self.setuper.conditional:
                vae = CVAE(self.num_res_type, self.setuper.dimensionality, self.len_protein * self.num_res_type,
                          self.setuper.layers)
            else:
                vae = VAE(self.num_res_type, self.setuper.dimensionality, self.len_protein * self.num_res_type,
                            self.setuper.layers)

            # random initialization of weights in the case of robustness
            if self.setuper.robustness_train:
                import torch.nn as nn
                from random import random, seed
                import time
                seed(time.time())
                spread = 0.15
                for en_layer, dec_layer in zip(vae.encoder_linears, vae.decoder_linears):
                    nn.init.normal_(en_layer.weight, mean=0.0, std=random() * spread)
                    nn.init.normal_(dec_layer.weight, mean=0.0, std=random() * spread)
                nn.init.normal_(vae.encoder_mu.weight, mean=0.0, std=random() * spread)
                nn.init.normal_(vae.encoder_logsigma.weight, mean=0.0, std=random() * spread)
                print("Train message : random weights init")

            # move the VAE onto a GPU
            if self.use_cuda:
                vae.cuda()

            # build the Adam optimizer
            optimizer = optim.Adam(vae.parameters(),
                                   weight_decay=self.setuper.decay)

            # collect training and validation data indices
            validation_idx = idx_subset[0] if self.setuper.robustness_train else idx_subset[k]
            validation_idx.sort()

            train_idx = np.array(list(set(range(self.num_seq)) - set(validation_idx)))
            train_idx.sort()

            solubility = None
            if self.setuper.conditional:
                solubility = torch.from_numpy(self.solubility[train_idx])
                if self.use_cuda:
                    solubility = solubility.cuda()

            train_msa = torch.from_numpy(self.seq_msa_binary[train_idx,])
            if self.use_cuda:
                train_msa = train_msa.cuda()

            train_weight = torch.from_numpy(self.seq_weight[train_idx])
            # train_weight = train_weight / torch.sum(train_weight) #Already done
            if self.use_cuda:
                train_weight = train_weight.cuda()

            train_data = MSA_Dataset(train_msa, train_weight, self.keys[train_idx], solubility)
            batch_size = 128 if self.setuper.convolution else train_msa.shape[0]
            train_data_loader = DataLoader(train_data, batch_size=batch_size)
            train_loss_list = []
            num_of_decay = self.setuper.epochs // 10
            dynamic_decay, decay_epoch_cnt = np.zeros(num_of_decay), 0
            dynamic_decay[:(num_of_decay//4)] = np.linspace(0.05 if self.setuper.decay == 0.0 else self.setuper.decay,
                                                            0, (num_of_decay//4))
            for epoch in range(self.setuper.epochs):
                train_loss_tmp = []
                for data in train_data_loader:
                    train_msa, train_weight, key, sol = data
                    if self.setuper.conditional:
                        loss = (-1) * vae.compute_weighted_elbo(train_msa, train_weight, self.setuper.C, c=sol)
                    else:
                        loss = (-1) * vae.compute_weighted_elbo(train_msa, train_weight, self.setuper.C)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (epoch+1) % 10 == 0 and self.setuper.dynamic_decay:
                        optimizer.param_groups[0]['weight_decay'] = dynamic_decay[decay_epoch_cnt]
                        print(" Decay weight value {}".format(optimizer.param_groups[0]['weight_decay']))
                        decay_epoch_cnt += 1
                    train_loss_tmp.append(loss.item())

                train_loss_list.extend(train_loss_tmp)
                print_limit = 5 if self.setuper.convolution else 50
                if (epoch + 1) % print_limit == 0:
                    print("Fold: {}, Epoch: {:>4}, loss: {:>4.2f}".format(k, epoch, train_loss_list[-1]), flush=True)

            ## cope trained model to cpu and save it
            if self.use_cuda:
                vae.cpu()

            # Save it to the special name
            model_name = self.setuper.VAE_model_dir + "/{}_fold_{}.model".format(self.setuper.model_name, k)
            torch.save(vae.state_dict(), model_name)

            print("Finish the {}th fold training".format(k))
            print(Helper.LOG_DELIMETER.value)
            print('')

            print("Start the {}th fold validation".format(k))
            print(Helper.LOG_DELIMETER.value)
            # evaluate the trained model
            if self.use_cuda:
                print("Cleaning CUDA cache")
                torch.cuda.empty_cache()
                vae.cuda()

            elbo_on_validation_data_list = []
            # because the function vae.compute_elbo_with_multiple samples uses
            # a large amount of memory on GPUs. we have to split validation data
            # into batches.
            batch_size = 20
            num_batches = len(validation_idx) // batch_size + 1
            if not self.setuper.convolution and not self.setuper.conditional:
                for idx_batch in range(num_batches):
                    if (idx_batch + 1) % 50 == 0:
                        print("idx_batch: {} out of {}".format(idx_batch, num_batches))
                    validation_msa = self.seq_msa_binary[
                        validation_idx[idx_batch * batch_size:(idx_batch + 1) * batch_size]]
                    validation_msa = torch.from_numpy(validation_msa)
                    with torch.no_grad():
                        if self.use_cuda:
                            validation_msa = validation_msa.cuda()
                        elbo = vae.compute_elbo_with_multiple_samples(validation_msa, 5000)
                        elbo_on_validation_data_list.append(elbo.cpu().data.numpy())

                elbo_on_validation_data = np.concatenate(elbo_on_validation_data_list)
                elbo_all_list.append(elbo_on_validation_data)

            print("Finish the {}th fold validation".format(k))
            print(Helper.LOG_DELIMETER.value)

            if self.use_cuda:
                torch.cuda.empty_cache()

            elbo_all = np.concatenate(elbo_all_list) if not self.setuper.convolution and \
                                                        not self.setuper.conditional else [0]
            elbo_mean = np.mean(elbo_all)
            # the mean_elbo can approximate the quanlity of the learned model
            # we want a model that has high mean_elbo
            # different weight decay factor or different network structure
            # will give different mean_elbo values and we want to choose the
            # weight decay factor or network structure that has large mean_elbo
            print("mean_elbo: {:.3f}".format(elbo_mean))

            with open(self.setuper.pickles_fld + "/elbo_all.pkl", 'wb') as file_handle:
                pickle.dump(elbo_all, file_handle)
            # In the case of robustness train save to file value with its final loss
            # to eliminate badly initiliazed models
            if self.setuper.robustness_train:
                filename = self.setuper.pickles_fld + '/ModelsRobustnessLosses.txt'
                if os.path.exists(filename):
                    append_write = 'a'  # append if already exists
                else:
                    append_write = 'w'  # make a new file if not

                hs = open(filename, append_write)
                hs.write("Model name,{},loss,{}\n".format(self.setuper.model_name, train_loss_list[-1]))
                hs.close()

    def load_pickles(self):
        with open(self.setuper.pickles_fld + "/seq_msa_binary.pkl", 'rb') as file_handle:
            seq_msa_binary = pickle.load(file_handle)
        with open(self.setuper.pickles_fld + "/seq_weight.pkl", 'rb') as file_handle:
            seq_weight = pickle.load(file_handle)
        seq_weight = seq_weight.astype(np.float32)
        with open(self.setuper.pickles_fld + "/keys_list.pkl", 'rb') as file_handle:
            seq_keys = pickle.load(file_handle)
        return seq_msa_binary, seq_weight, seq_keys


if __name__ == '__main__':
    tar_dir = CmdHandler()
    Downloader(tar_dir)
    if not tar_dir.robustness_train:
        Train(tar_dir, benchmark=True).train()
    else:
        Train(tar_dir, benchmark=True).train()
