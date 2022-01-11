__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/28 13:49:00"

import pickle
import numpy as np
import torch
import torch.optim as optim

from parser_handler import CmdHandler
from download_MSA import Downloader
from VAE_model import MSA_Dataset, VAE
from matplotlib import pyplot as plt

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

        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

        percentage = 20
        self.benchmark = benchmark
        self.benchmark_set = np.zeros((self.num_seq // percentage, self.len_protein, self.num_res_type))
        if benchmark:
            # Take 5 percent from original MSA for further evaluation
            random_idx = np.random.permutation(range(self.num_seq))
            for i in range(self.num_seq // percentage):
                self.benchmark_set[i] = self.seq_msa_binary[random_idx[i]]
            self.seq_msa_binary = np.delete(self.seq_msa_binary, random_idx[:(self.num_seq // percentage)], axis=0)
            self.num_seq = self.seq_msa_binary.shape[0]
            with open(setuper.pickles_fld + '/positive_control.pkl', 'wb') as file_handle:
                pickle.dump(self.benchmark_set, file_handle)
            with open(setuper.pickles_fld + '/training_set.pkl', 'wb') as file_handle:
                pickle.dump(self.seq_msa_binary, file_handle)
        self.seq_msa_binary = self.seq_msa_binary.reshape((self.num_seq, -1))
        self.seq_msa_binary = self.seq_msa_binary.astype(np.float32)

    def train(self):
        ## Split data into 6 subsets
        num_subsets = 5
        num_seq_subset = self.num_seq // num_subsets + 1
        idx_subset = []
        random_idx = np.random.permutation(range(self.num_seq))
        for i in range(num_subsets):
            idx_subset.append(random_idx[i * num_seq_subset:(i + 1) * num_seq_subset])

        ## the following list holds the elbo values on validation data
        elbo_all_list = []

        print("Start the validation fold training")
        print("-" * 60)

        ## build a VAE model with random parameters
        vae = VAE(self.num_res_type, self.setuper.dimensionality, self.len_protein * self.num_res_type, self.setuper.layers)

        ## move the VAE onto a GPU
        if self.use_cuda:
            vae.cuda()

        ## build the Adam optimizer
        optimizer = optim.Adam(vae.parameters(),
                               weight_decay=self.setuper.decay)

        ## collect training and valiation data indices
        validation_idx = idx_subset[num_subsets-1]
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

        elbo_on_validation_data_list = []
        ## because the function vae.compute_elbo_with_multiple samples uses
        ## a large amount of memory on GPUs. we have to split validation data
        ## into batches.
        batch_size = 32
        num_batches = len(validation_idx) // batch_size + 1
        train_loss_list = []
        validation_loss_list = []

        epoch = 0
        epoch_cond_list = 3 * [False]
        last_progress_elbo = float("inf")
        errors_accepted = 100 # Take last 100 samples from array epochs
        err_frac = 0.8 # 80 percent error tolerance along last 100 validations

        # More than 80 % (80 iterations) of last validations have to be worse than
        # the last successful learning step to quit validation and determine correct
        while not (sum(epoch_cond_list[-errors_accepted:]) > err_frac * errors_accepted):
            loss = (-1) * vae.compute_weighted_elbo(train_msa, train_weight, self.setuper.C)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                train_loss_list.append(loss.item())

                for idx_batch in range(num_batches):
                    if ((idx_batch + 1) % 50 == 0) and ((epoch + 1) % 50 == 0):
                        print("idx_batch: {} out of {}".format(idx_batch, num_batches))
                    validation_msa = self.seq_msa_binary[
                        validation_idx[idx_batch * batch_size:(idx_batch + 1) * batch_size]]
                    validation_msa = torch.from_numpy(validation_msa)
                    with torch.no_grad():
                        if self.use_cuda:
                            validation_msa = validation_msa.cuda()
                        elbo = (-1) * vae.compute_elbo_with_multiple_samples(validation_msa, 3000)
                        elbo_on_validation_data_list.append(elbo.cpu().data.numpy())

                elbo_on_validation_data = np.concatenate(elbo_on_validation_data_list)
                elbo_all_list.append(elbo_on_validation_data)

                elbo_all = np.concatenate(elbo_all_list)
                elbo_mean = np.mean(elbo_all)

                validation_loss_list.append(elbo_mean)

                if (epoch + 1) % 50 == 0:
                    print("Epoch: {:>4}, loss: {:>4.2f}, validation loss: {:>4.2f}".format(epoch, loss.item(), elbo_mean), flush=True)

                val_deviation, last_progress_elbo = (True, last_progress_elbo) if elbo_mean > last_progress_elbo \
                                                                                    else (False, elbo_mean)
                epoch_cond_list.append(val_deviation)
            epoch += 1

        ## cope trained model to cpu and save it
        if self.use_cuda:
            vae.cpu()
        torch.save(vae.state_dict(), self.setuper.VAE_model_dir + "/vae_{}_fold_{}_C_{}_D_{}_{}.model".format(
            str(self.setuper.decay), 10, str(self.setuper.C), str(self.setuper.dimensionality),
            self.setuper.layersString))

        ## Plot graph
        save_path = self.setuper.high_fld + '/trainingValidation_epochs_{}-{}.png'.format(epoch, errors_accepted)

        plt.plot(train_loss_list)
        plt.plot(validation_loss_list)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        print("generating training validation plot into", save_path)
        plt.savefig(save_path)

        print("Finish the validation fold training")
        print("=" * 60)
        print('')

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
    tar_dir.setup_struct()
    Downloader(tar_dir)
    Train(tar_dir, benchmark=True).train()