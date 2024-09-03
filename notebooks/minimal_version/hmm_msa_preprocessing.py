import os
import sys

import numpy as np
import pandas as pd
import subprocess
from Bio import SeqIO, AlignIO
from pathlib import Path

from notebooks.minimal_version.latent_space import LatentSpace
from notebooks.minimal_version.parser_handler import RunSetup
from notebooks.minimal_version.msa import MSA
from notebooks.minimal_version.utils import store_to_fasta, load_from_pkl


def fix_seqs(input_string):
    seq_ = input_string.replace('.', '')
    seq_ = ''.join([char for char in seq_ if not char.islower()])
    return seq_


class HMMAligner:
    def __init__(self, run: RunSetup, hmmer_custom_path=""):

        self.build = os.path.join(hmmer_custom_path, 'hmmbuild')
        self.align = os.path.join(hmmer_custom_path, 'hmmalign')
        if not os.path.exists(self.build) and hmmer_custom_path != "":
            print(f"There is no HMM binary in path {self.build}, quitting")
            sys.exit(1)
        if not os.path.exists(self.align) and hmmer_custom_path != "":
            print(f"There is no HMM binary in path {self.align}, quitting")
            sys.exit(1)

        print(f"Using hmmer binaries from {hmmer_custom_path} directory")

        self.run = run

        self.hmmer_path = hmmer_custom_path
        attr_dir = os.path.join(run.root_dir, "hmm")
        os.makedirs(attr_dir, exist_ok=True)
        self.hmm_result = attr_dir
        print(f"Creating HMM directory in {attr_dir}")
        self.hmm_model = os.path.join(self.hmm_result, 'hmmModel.hmm')


    def buildHMM(self, input_msa=None):
        print('Building HMM model...')
        if input_msa is None:
            input_msa = self.run.dataset
        print(f"Using msa from {input_msa}")

        log_output = os.path.join(self.run.logs, 'hmmModel_hmmbuild.log')
        subprocess.run(f'{self.build} --amino {self.hmm_model} {input_msa} > {log_output}',shell=True)
        print(f"HMM model built into {self.hmm_model}")

        print(f"Creating raw fasta file from {input_msa} MSA (removing gap symbols)...)")
        msa = MSA.load_msa(input_msa)
        raw_fasta = {k: seq.replace("-", "") for k, seq in msa.items()}
        raw_fasta = {k: seq.replace(".", "") for k, seq in raw_fasta.items()}

        msa_path = Path(input_msa)
        raw_fasta_name = msa_path.name.split('.')[0] + "_raw.fa"
        raw_fasta_path = os.path.join(msa_path.parent, raw_fasta_name)
        store_to_fasta(raw_fasta, raw_fasta_path)
        print(f"Prepared raw fasta file to be realign to hmm profile into {raw_fasta_path}")

        hmm_fasta_name = msa_path.name.split('.')[0] + "_hmm.fa"
        hmm_fasta_path = os.path.join(msa_path.parent, hmm_fasta_name)

        subprocess.run(
            f'cat {raw_fasta_path} | {self.align} --trim --outformat afa {self.hmm_model} - > {hmm_fasta_path}',
            shell=True)

        hmm_msa = AlignIO.read(f'{hmm_fasta_path}', 'fasta')

        fix_hmm_fasta_name = msa_path.name.split('.')[0] + "_hmm_fix.fa"
        fix_hmm_fasta_path = os.path.join(msa_path.parent, fix_hmm_fasta_name)

        with open(fix_hmm_fasta_path, 'w') as out_file:
            for prot in hmm_msa:
                out_file.write('>' + prot.id + '\n')
                out_file.write(fix_seqs(str(prot.seq)))
                out_file.write('\n')

        print(f"HMM MSA realigned and prepared into {fix_hmm_fasta_path}")

        self.run.dataset = fix_hmm_fasta_path
        print(f"Dataset variable changed in current RunSetup instance, to keep it consistent, change the dataset field in you config to {fix_hmm_fasta_path}")

    def encode_custom_seqs(self, fasta_path, model=f'vae_fold_0.model', batch_size=1):
        afa_path = self.hmmer_align(fasta_path)
        df = self._aligned_df(afa_path)
        self.run.weights = os.path.join(self.run.root_dir, "model", model)
        latent_space = LatentSpace(self.run)
        mu1_ = []
        mu2_ = []
        for i in range(0, len(df), batch_size):
            latent_embeddings = latent_space.encode(df['trimmed_afa'].to_list()[i:i + batch_size])[0]
            mu1_.append(latent_embeddings[:, 0])
            mu2_.append(latent_embeddings[:, 1])
        df['mu1'] = np.concatenate(mu1_)
        df['mu2'] = np.concatenate(mu2_)
        return df

    def _aligned_df(self, fasta_path):
        msa_columns_path = os.path.join(self.run.pickles, "msa_columns.pkl")
        if not os.path.exists(msa_columns_path):
            print("Please MSA process sequences first!")
            sys.exit(2)

        msa_columns = load_from_pkl(msa_columns_path)
        dict = {seq.description: fix_seqs(str(seq.seq)) for seq in SeqIO.parse(fasta_path, "fasta")}
        df = pd.DataFrame.from_dict(dict, orient='index', columns=['afa'])
        df['trimmed_afa'] = df['afa'].apply(lambda x: ''.join([x[i] for i in msa_columns]))
        return df

    def hmmer_align(self, fasta_path):
        fasta_name = Path(fasta_path).name
        output_path = os.path.join(self.hmm_result, fasta_name)
        subprocess.run(
            f'{self.align} --trim --outformat afa {self.hmm_model} {fasta_path} >> {output_path}',
            shell=True)
        return output_path
