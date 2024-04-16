__author__ = "Pavel Kohout <xkohou15@vutbr.cz>"
__date__ = "2023/01/24 14:40:00"

import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from VAE_model import VAE, MSA_Dataset
from parser_handler import CmdHandler
from project_enums import Helper
from vae_models.cnn_vae import VaeCnn
from vae_models.conditional_vae import CVAE


class TrainEnsemble:
    def __init__(self, cmd_liner: CmdHandler, model_num: int):
        self.cmd = cmd_liner
        self.dst = cmd_liner.VAE_model_dir
        self.model_num = model_num
        self.ens_dir = cmd_liner.pickles_fld + "/ensembles/"
        self.init_msa_meta_data()

        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

    def train(self):
        cmd = self.cmd

        # build a VAE model with random parameters
        if cmd.convolution:
            vae = VaeCnn(cmd.dimensionality, self.len_protein)
        elif cmd.conditional:
            vae = CVAE(self.num_res_type,
                       cmd.dimensionality,
                       self.len_protein * self.num_res_type,
                       cmd.layers)
        else:
            vae = VAE(self.num_res_type,
                      cmd.dimensionality,
                      self.len_protein * self.num_res_type,
                      cmd.layers)

        # move the VAE onto a GPU
        if self.use_cuda:
            vae.cuda()

        # build the Adam optimizer
        optimizer = optim.Adam(vae.parameters(),
                               weight_decay=cmd.decay)

        solubility = None
        if cmd.conditional:
            solubility = torch.from_numpy(self.solubility_set)
            if self.use_cuda:
                solubility = solubility.cuda()

        train_msa = torch.from_numpy(self.training_set)
        if self.use_cuda:
            train_msa = train_msa.cuda()

        train_weight = torch.from_numpy(self.seq_weight)
        # train_weight = train_weight / torch.sum(train_weight) #Already done
        if self.use_cuda:
            train_weight = train_weight.cuda()

        train_data = MSA_Dataset(train_msa, train_weight, self.seq_keys, solubility)
        batch_size = 128 if cmd.convolution else train_msa.shape[0]
        train_data_loader = DataLoader(train_data, batch_size=batch_size)
        train_loss_list = []

        # Decrease L2 regularization term influence over time
        num_of_decay = cmd.epochs // 10
        dynamic_decay, decay_epoch_cnt = np.zeros(num_of_decay), 0
        dynamic_decay[:(num_of_decay // 4)] = np.linspace(0.05 if cmd.decay == 0.0 else cmd.decay,
                                                          0, (num_of_decay // 4))

        # train our model
        for epoch in range(cmd.epochs):
            train_loss_tmp = []
            for data in train_data_loader:
                train_msa, train_weight, key, sol = data
                if cmd.conditional:
                    loss = (-1) * vae.compute_weighted_elbo(train_msa, train_weight, cmd.C, c=sol)
                else:
                    loss = (-1) * vae.compute_weighted_elbo(train_msa, train_weight, cmd.C)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # modify weight decay influence every 10th epoch
                if (epoch + 1) % 10 == 0 and cmd.dynamic_decay:
                    optimizer.param_groups[0]['weight_decay'] = dynamic_decay[decay_epoch_cnt]
                    print(" Decay weight value {}".format(optimizer.param_groups[0]['weight_decay']))
                    decay_epoch_cnt += 1
                train_loss_tmp.append(loss.item())

            train_loss_list.extend(train_loss_tmp)
            print_limit = 5 if cmd.convolution else 50
            if (epoch + 1) % print_limit == 0:
                print("Epoch: {:>4}, loss: {:>4.2f}".format(epoch, train_loss_list[-1]), flush=True)

        ## cope trained model to cpu and save it
        if self.use_cuda:
            vae.cpu()

        # Save it to the special name
        model_name = cmd.VAE_model_dir + "/{}_ensemble_{}.model".format(cmd.model_name, self.model_num)
        torch.save(vae.state_dict(), model_name)

        print("Finish the model training")
        print(Helper.LOG_DELIMETER.value)
        print('')

        print("Start the model validation")
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
        num_batches = self.validation_set.shape[0] // batch_size + 1
        if not cmd.convolution and not cmd.conditional:
            for idx_batch in range(num_batches):
                if (idx_batch + 1) % 50 == 0:
                    print("idx_batch: {} out of {}".format(idx_batch, num_batches))
                validation_msa = self.training_set[idx_batch * batch_size:(idx_batch + 1) * batch_size]
                validation_msa = torch.from_numpy(validation_msa)
                with torch.no_grad():
                    if self.use_cuda:
                        validation_msa = validation_msa.cuda()
                    elbo = vae.compute_elbo_with_multiple_samples(validation_msa, 5000)
                    elbo_on_validation_data_list.append(elbo.cpu().data.numpy())

            elbo_on_validation_data = np.concatenate(elbo_on_validation_data_list)

        print("Finish the model validation")
        print(Helper.LOG_DELIMETER.value)

        if self.use_cuda:
            torch.cuda.empty_cache()

        elbo_all = np.concatenate(elbo_on_validation_data) if not cmd.convolution and \
                                                              not cmd.conditional else [0]
        print("ELBO value: {:.3f}".format(elbo_all[0]))

    def init_msa_meta_data(self):
        """ Load data corresponding to this model """
        ens_dir = self.ens_dir
        try:
            with open(ens_dir + f"/training_set{self.model_num}.pkl", 'rb') as file_handle:
                self.training_set = pickle.load(file_handle)
            with open(ens_dir + f"/validation_set{self.model_num}.pkl", 'rb') as file_handle:
                self.validation_set = pickle.load(file_handle)
            with open(ens_dir + f"/solubility_set{self.model_num}.pkl", 'rb') as file_handle:
                self.solubility_set = pickle.load(file_handle)
            with open(ens_dir + "/training_weights.pkl", 'rb') as file_handle:
                self.seq_weight = pickle.load(file_handle).astype(np.float32)
            with open(ens_dir + "/training_keys.pkl", 'rb') as file_handle:
                self.seq_keys = np.array(pickle.load(file_handle))
        except FileNotFoundError:
            print("\n\tERROR please run MSA preprocessing for ensembles"
                  "\n\t\tpython3 runner.py msa_handlers/ensemble_preprocess.py --json [your_config_path] "
                  "\n\tbefore running this command")
            exit(1)

        # Set all meta information
        self.num_seq = self.training_set.shape[0]
        self.len_protein = self.training_set.shape[1]
        self.num_res_type = self.training_set.shape[2]


if __name__ == '__main__':
    cmd = CmdHandler()
    trainer = TrainEnsemble(cmd_liner=cmd, model_num=cmd.ens_num)
