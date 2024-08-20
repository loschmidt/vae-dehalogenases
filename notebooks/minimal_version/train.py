__author__ = "Pavel Kohout <xkohou15@vutbr.cz>"
__date__ = "2020/06/26 13:49:00"

import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# my libraries
from notebooks.minimal_version.parser_handler import RunSetup
from notebooks.minimal_version.utils import load_from_pkl, store_to_pkl
from notebooks.minimal_version.models.model_dataloader import MsaDataset
from notebooks.minimal_version.models.ai_models import AIModel


def setup_train(run: RunSetup, model: str):
    """
    Prepare model hyperparameters, training and evaluation dataset
    :param run: object with parsed run parameters
    :return:
    """
    msa_binary = load_from_pkl(os.path.join(run.pickles, "seq_msa_binary.pkl"))
    keep_keys = load_from_pkl(os.path.join(run.pickles, "keep_keys.pkl"))  # all evo queries
    evo_cnt = len(keep_keys)

    num_seq = msa_binary.shape[0]
    len_protein = msa_binary.shape[1]
    num_res_type = msa_binary.shape[2]

    random_idx = None

    if run.run_capacity_test:
        num_seq_capacity = int(num_seq // (100 / run.percentage_of_seq_for_evaluation))
        print(f" Leaving out {run.percentage_of_seq_for_evaluation}% sequences from training "
              f"for generative capacity evaluation ({num_seq_capacity} sequences in total)")
        random_idx = np.random.permutation(range(evo_cnt, num_seq))  # do not include fixed sequences into evaluation
        benchmark_indices = random_idx[:num_seq_capacity]
        benchmark_set = msa_binary[benchmark_indices]
        msa_binary = np.delete(msa_binary, benchmark_indices, axis=0)
        num_seq = msa_binary.shape[0]
        store_to_pkl(benchmark_set, os.path.join(run.pickles, "positive_control.pkl"))
        store_to_pkl(benchmark_indices, os.path.join(run.pickles, "positive_control_ids.pkl"))
        # update random index variable for further processing use, make fixed sequences first!
        random_idx = np.append(np.array(range(0, evo_cnt)), random_idx[num_seq_capacity:])

    # model selection
    model_param = {
        "hidden_units": [len_protein] if run.layers is None else run.layers,
        "lat_dim": run.lat_dim,
        "in_out_size": len_protein * num_res_type,
        "aa_count": num_res_type,
        "model": model,
        "label_dim": 0
    }
    store_to_pkl(model_param, os.path.join(run.model, "model_params.pkl"))
    # training subset
    num_seq_subset = num_seq // 5  # % of sequences just use for evaluation

    idx_subsets = []
    random_idx = random_idx if random_idx is not None else np.random.permutation(range(evo_cnt, num_seq))  # evo queries are the first X sequences

    for i in range(5):
        idx_subsets.append(random_idx[i * num_seq_subset:(i + 1) * num_seq_subset])

    # We do not create additional training pickles wi will stick to the indexes for training
    idx_subsets[0] = np.append(idx_subsets[0], range(evo_cnt))  # append all queries into the first subset of indices
    store_to_pkl(idx_subsets, os.path.join(run.pickles, "idx_subsets.pkl"))


def check_loss_progress(loss_list, current_epoch, epochs) -> bool:
    """
    Flag for checking the last 10 epochs and progress of loss value.
    End if there is no better loss for the last 10 epochs.
    Also check whether this control is appropriate, according to the configuration settings (given epoch parameter):
    :param loss_list: list, with last lost values
    :param current_epoch: number of current epoch
    :param epochs: if None is specified run training until no progress reported otherwise run given number of
    epochs
    :return: bool value true if training should continue
    """
    if epochs is not None:  # there is a given number of epochs
        return current_epoch < epochs
    if current_epoch < 20:  # too less numb of iteration
        return True
    prev_loss = loss_list[-11]
    for loss in loss_list[-10:]:
        if loss < prev_loss:  # we minimizing the value
            return True
    return False


class Train:
    def __init__(self, run: RunSetup):
        self.run = run
        self.model_param = load_from_pkl(os.path.join(run.model, "model_params.pkl"))

        self.ai_model = AIModel(self.model_param)
        self.K = self.run.K  # train just one model, but be ready for ensemble extension!!!
        assert self.K < 5

        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

        msa_binary = load_from_pkl(os.path.join(run.pickles, "seq_msa_binary.pkl"))
        self.seq_msa_binary = msa_binary.reshape((msa_binary.shape[0], -1))
        self.seq_msa_binary = self.seq_msa_binary.astype(np.float32)
        self.seq_weight = load_from_pkl(os.path.join(run.pickles, "seq_weight.pkl"))

    def train(self):
        idx_subset = load_from_pkl(os.path.join(self.run.pickles, "idx_subsets.pkl"))
        sets_count = len(idx_subset)

        # the following list holds the elbo values on validation data
        elbo_all_list = []

        _, conditional_model = self.ai_model.get_model()
        all_labels, label_dim = None, 0
        if conditional_model:
            print(f"Conditional model - loading labels from {self.run.conditional_data}")
            all_labels = load_from_pkl(os.path.join(self.run.conditional_data, "conditional_labels_to_categories.pkl"))
            # update model params
            model_params = load_from_pkl(os.path.join(self.run.model, "model_params.pkl"))
            model_params["label_dim"] = all_labels[0].shape[0]  # it should be one dim vector with categories
            store_to_pkl(model_params, os.path.join(self.run.model, "model_params.pkl"))
            self.ai_model = AIModel(model_params)  # update instance

        # Count of validations run is for robustness just one
        for k in range(self.K):
            print("Start the {}th fold training".format(k))
            print("-" * 60)
            vae, _ = self.ai_model.get_model()
            if self.use_cuda:  # move the VAE onto a GPU
                vae.cuda()

            # build the Adam optimizer
            optimizer = optim.Adam(vae.parameters(),
                                   weight_decay=self.run.decay)

            # collect training and validation data indices
            validation_idx = idx_subset[sets_count-(k+1)]
            validation_idx.sort()

            train_idx = np.array(list(set(range(self.seq_msa_binary.shape[0])) - set(validation_idx)))
            train_idx.sort()

            train_labels = None
            if conditional_model:
                train_labels = all_labels[train_idx]
                if self.use_cuda:
                    train_labels = train_labels.cuda()

            train_msa = torch.from_numpy(self.seq_msa_binary[train_idx,])
            if self.use_cuda:
                train_msa = train_msa.cuda()

            train_weight = torch.from_numpy(self.seq_weight[train_idx])
            if self.use_cuda:
                train_weight = train_weight.cuda()

            train_data = MsaDataset(train_msa, train_weight, train_labels)
            batch_size = train_msa.shape[0] if self.run.batch_size is None else self.run.batch_size
            train_data_loader = DataLoader(train_data, batch_size=batch_size)
            train_loss_list = []

            # there is dynamic decay required, setup decay linear decrease for 1st quarter of epochs or for 1000
            num_of_decay = self.run.epochs if self.run.epochs is not None else 4000
            dynamic_decay = np.zeros(num_of_decay)
            dynamic_decay[:(num_of_decay // 4)] = np.linspace(0.05 if self.run.decay == 0.0 else self.run.decay,
                                                              0, (num_of_decay // 4))

            epoch = 0
            while check_loss_progress(train_loss_list, epoch, self.run.epochs):
                train_loss_tmp = []
                for data in train_data_loader:
                    train_msa, train_weight, batch_labels = data
                    if not conditional_model:
                        batch_labels = None
                    loss = (-1) * vae.compute_weighted_elbo(train_msa, train_weight, c=batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss_tmp.append(loss.item())
                if self.run.dynamic_decay:
                    if dynamic_decay.shape[0] > epoch and (epoch+1) % 10 == 0:  # change decay every 10th epoch
                        optimizer.param_groups[0]['weight_decay'] = dynamic_decay[epoch]

                train_loss_list.extend(train_loss_tmp)
                if (epoch + 1) % 50 == 0:
                    print("Fold: {}, Epoch: {:>4}, loss: {:>4.2f} ".format(k, epoch, train_loss_list[-1]), flush=True,
                          end=" ")
                    print("Decay weight values {:>4.5f}".format(optimizer.param_groups[0]['weight_decay']))
                epoch += 1

            # cope trained model to cpu and save it
            if self.use_cuda:
                vae.cpu()

            # Save it to the special name
            model_name = os.path.join(self.run.model, f"vae_fold_{k}.model")
            torch.save(vae.state_dict(), model_name)
            print(f" Training VAE model store into {model_name}")

            print("Finish the {}th fold training".format(k))
            print("Start the {}th fold validation".format(k))
            # evaluate the trained model
            if self.use_cuda:
                print("Cleaning CUDA cache")
                torch.cuda.empty_cache()
                vae.cuda()

            elbo_on_validation_data_list = []
            # because the function vae.compute_elbo_with_multiple samples uses
            # a large amount of memory on GPUs. we have to split validation data
            # into batches.
            batch_size = 10
            num_batches = len(validation_idx) // batch_size + 1
            for idx_batch in range(num_batches):
                if (idx_batch + 1) % 50 == 0:
                    print("idx_batch: {} out of {}".format(idx_batch, num_batches))
                validation_msa = self.seq_msa_binary[
                    validation_idx[idx_batch * batch_size:(idx_batch + 1) * batch_size]]
                validation_msa = torch.from_numpy(validation_msa)
                validation_labels = None
                if conditional_model:
                    validation_labels = all_labels[
                        validation_idx[idx_batch * batch_size:(idx_batch + 1) * batch_size]]
                with torch.no_grad():
                    if self.use_cuda:
                        validation_msa = validation_msa.cuda()
                        if conditional_model:
                            validation_labels = validation_labels.cuda()
                        else:
                            validation_labels = None
                    elbo = vae.compute_elbo_with_multiple_samples(validation_msa, 5000, c=validation_labels)
                    elbo_on_validation_data_list.append(elbo.cpu().data.numpy())

            elbo_on_validation_data = np.concatenate(elbo_on_validation_data_list)
            elbo_all_list.append(elbo_on_validation_data)

            print("Finish the {}th fold validation".format(k))
