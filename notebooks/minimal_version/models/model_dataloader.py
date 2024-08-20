from torch.utils.data import Dataset


class MsaDataset(Dataset):
    """
    Dataset class for multiple sequence alignment.
    """

    def __init__(self, seq_msa_binary, seq_weight, conditional_labels=None):
        """
        seq_msa_binary: a two-dimensional np.array.
                        size: [num_of_sequences, length_of_msa*num_amino_acid_types]
        seq_weight: one dimensional array.
                    size: [num_sequences].
                    Weights for sequences in a MSA.
                    The sum of seq_weight has to be equal to 1 when training latent space models using VAE
        seq_weight: array of tensors with one/multiple categories.
                    size: [num_sequences, number of categories].
        """
        super(MsaDataset).__init__()
        self.seq_msa_binary = seq_msa_binary
        self.seq_weight = seq_weight

        # provide tmp class which wil be not used
        self.conditional_labels = conditional_labels if conditional_labels is not None else [0 for _ in range(len(seq_msa_binary))]

    def __len__(self):
        assert (self.seq_msa_binary.shape[0] == len(self.seq_weight))
        assert (len(self.conditional_labels) == len(self.seq_weight))
        return self.seq_msa_binary.shape[0]

    def __getitem__(self, idx):
        return self.seq_msa_binary[idx, :], self.seq_weight[idx], self.conditional_labels[idx]
