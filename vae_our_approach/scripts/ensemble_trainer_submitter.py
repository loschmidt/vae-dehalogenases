__author__ = "Pavel Kohout <xkohou15@vutbr.cz>"
__date__ = "2023/01/25 14:10:00"

import os

from parser_handler import CmdHandler

# Parser options passed via command line
cmd_liner = CmdHandler()

# Get current configuration file for experiment shared by all ensemble models
conf_file_path = os.environ["VAE_CONF"]

# Get folder with preprocessed MSAs and data splits for individual models in ensemble
pkl_path = cmd_liner.pickles_fld[5:]  # remove characters ./../results... in script path

for model_i in range(cmd_liner.ens_cnt):
    cmd = f'qsub -N ens_tr{model_i} -v conf_path="{conf_file_path}", pkl_fld="{pkl_path}", model="{model_i}" ' \
          f'./../pbs_scripts/ensemble_trainer.sh'
    print(f"\tSubmitted job ens_tr{model_i} by running command ", cmd)
    os.system(cmd)
