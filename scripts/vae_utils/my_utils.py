
def acc_to_mu(embedding_dict: dict, acc_list: list):
    """
    Get latent space mu corresponding to the given accessions
    @WARNING non-existing keys are returned as the third variable, check it as you wish on the output
    return: (mus, mu_keys, non-mapped-indices-list)
    """
    accessions = embedding_dict['keys']
    keys, mus = [], []
    non_mapped = []
    for acc in acc_list:
        try:
            index = accessions.index(acc)
            mus.append(embedding_dict['mu'][index])
            keys.append(acc)
        except:
            non_mapped.append(acc)
    return mus, keys, non_mapped
