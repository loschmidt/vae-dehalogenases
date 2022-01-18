__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/28 13:49:00"

import os
import pickle

import numpy as np
import torch
import torch.optim as optim

from VAE_model import VAE
from download_MSA import Downloader
from parser_handler import CmdHandler


class Train:
    def __init__(self, setuper: CmdHandler, msa=None, benchmark=False):
        self.setuper = setuper
        if msa is None:
            ## Run just train script, pickles files should be ready load them
            self._load_pickles()
        else:
            self.seq_msa_binary = msa.seq_msa_binary
            self.seq_weight = msa.seq_weight
            self.seq_keys = msa.keys_list
        self.num_seq = self.seq_msa_binary.shape[0]
        self.len_protein = self.seq_msa_binary.shape[1]
        self.num_res_type = self.seq_msa_binary.shape[2]

        self.K = setuper.K

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
            for i in range(self.num_seq // percentage):
                self.benchmark_set[i] = self.seq_msa_binary[random_idx[i]]
            self.seq_msa_binary = np.delete(self.seq_msa_binary, random_idx[:(self.num_seq // percentage)], axis=0)
            training_weights = np.delete(self.seq_weight, random_idx[:(self.num_seq // percentage)], axis=0)
            training_keys = np.delete(self.seq_keys, random_idx[:(self.num_seq // percentage)], axis=0)
            self.num_seq = self.seq_msa_binary.shape[0]
            with open(setuper.pickles_fld + '/positive_control.pkl', 'wb') as file_handle:
                pickle.dump(self.benchmark_set, file_handle)
            with open(setuper.pickles_fld + '/training_set.pkl', 'wb') as file_handle:
                pickle.dump(self.seq_msa_binary, file_handle)
            with open(setuper.pickles_fld + '/training_keys.pkl', 'wb') as file_handle:
                pickle.dump(training_keys, file_handle)
            with open(setuper.pickles_fld + '/training_weights.pkl', 'wb') as file_handle:
                pickle.dump(training_weights, file_handle)
        self.seq_msa_binary = self.seq_msa_binary.reshape((self.num_seq, -1))
        self.seq_msa_binary = self.seq_msa_binary.astype(np.float32)

    def train(self):
        num_seq_subset = self.num_seq // self.K + 1
        idx_subset = []
        random_idx = np.random.permutation(range(self.num_seq))
        for i in range(self.K):
            idx_subset.append(random_idx[i * num_seq_subset:(i + 1) * num_seq_subset])

        ## the following list holds the elbo values on validation data
        elbo_all_list = []

        # Count of validations run is for robustness just one
        K = 1 if self.setuper.robustness_train else self.K
        K = 1        
        for k in range(K):
            print("Start the {}th fold training".format(k))
            print("-" * 60)

            # build a VAE model with random parameters
            vae = VAE(self.num_res_type, self.setuper.dimensionality, self.len_protein * self.num_res_type, self.setuper.layers)

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
            validation_idx = idx_subset[k]
            validation_idx.sort()

            train_idx = np.array(list(set(range(self.num_seq)) - set(validation_idx)))
            train_idx.sort()

            train_msa = torch.from_numpy(self.seq_msa_binary[train_idx,])
            if self.use_cuda:
                train_msa = train_msa.cuda()

            train_weight = torch.from_numpy(self.seq_weight[train_idx])
            #train_weight = train_weight / torch.sum(train_weight) #Already done
            if self.use_cuda:
                train_weight = train_weight.cuda()

            train_loss_list = []
            for epoch in range(self.setuper.epochs):
                loss = (-1) * vae.compute_weighted_elbo(train_msa, train_weight, self.setuper.C)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_list.append(loss.item())
                if (epoch + 1) % 50 == 0:
                    print("Fold: {}, Epoch: {:>4}, loss: {:>4.2f}".format(k, epoch, loss.item()), flush=True)

            ## cope trained model to cpu and save it
            if self.use_cuda:
                vae.cpu()

            # if self.setuper.robustness_train:
            #     # Save it to the special name
            #     torch.save(vae.state_dict(), self.setuper.VAE_model_dir + "/{}.model".format(self.setuper.model_name))
            # else:
            #     torch.save(vae.state_dict(), self.setuper.VAE_model_dir + "/vae_{}_fold_{}_C_{}_D_{}_{}.model".format(
            #         str(self.setuper.decay), k, str(self.setuper.C), str(self.setuper.dimensionality),
            #         self.setuper.layersString))
            # Save it to the special name
            model_name = self.setuper.VAE_model_dir + "/{}_fold_{}.model".format(self.setuper.model_name, k)
            torch.save(vae.state_dict(), model_name)

            print("Finish the {}th fold training".format(k))
            print("=" * 60)
            print('')

            print("Start the {}th fold validation".format(k))
            print("-" * 60)
            ## evaluate the trained model
            if self.use_cuda:
                print("Cleaning CUDA cache")
                torch.cuda.empty_cache()
                vae.cuda()

            elbo_on_validation_data_list = []
            ## because the function vae.compute_elbo_with_multiple samples uses
            ## a large amount of memory on GPUs. we have to split validation data
            ## into batches.
            batch_size = 20
            num_batches = len(validation_idx) // batch_size + 1
            for idx_batch in range(num_batches):
                if (idx_batch + 1) % 50 == 0:
                    print("idx_batch: {} out of {}".format(idx_batch, num_batches))
                validation_msa = self.seq_msa_binary[validation_idx[idx_batch * batch_size:(idx_batch + 1) * batch_size]]
                validation_msa = torch.from_numpy(validation_msa)
                with torch.no_grad():
                    if self.use_cuda:
                        validation_msa = validation_msa.cuda()
                    elbo = vae.compute_elbo_with_multiple_samples(validation_msa, 5000)
                    elbo_on_validation_data_list.append(elbo.cpu().data.numpy())

            elbo_on_validation_data = np.concatenate(elbo_on_validation_data_list)
            elbo_all_list.append(elbo_on_validation_data)

            print("Finish the {}th fold validation".format(k))
            print("=" * 60)

            if self.use_cuda:
                torch.cuda.empty_cache()

            elbo_all = np.concatenate(elbo_all_list)
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

    def _load_pickles(self):
        with open(self.setuper.pickles_fld + "/seq_msa_binary.pkl", 'rb') as file_handle:
            self.seq_msa_binary = pickle.load(file_handle)
        with open(self.setuper.pickles_fld + "/seq_weight.pkl", 'rb') as file_handle:
            seq_weight = pickle.load(file_handle)
        self.seq_weight = seq_weight.astype(np.float32)
        with open(self.setuper.pickles_fld + "/keys_list.pkl", 'rb') as file_handle:
            self.seq_keys = pickle.load(file_handle)

if __name__ == '__main__':
    tar_dir = CmdHandler()
    Downloader(tar_dir)
    if not tar_dir.robustness_train:
        Train(tar_dir, benchmark=True).train()
    else:
        Train(tar_dir, benchmark=False).train()
