__author__ = "Pavel Kohout xkohou15@vutbr.cz"
__date__ = "2023/02/06"
__description__ = "MSA preprocessing step for the projection of of dataset into the latent space while perceiving as " \
                  "many sequences as possible. " \
                  "Does not apply classical steps of the MSA preprocessing that reduce the number of sequences " \
                  "significantly. Only sequences with low MSA sequential overlap with query are excluded (<50%)"

import inspect
import os
import sys

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0, parentDir)

from parser_handler import CmdHandler
from msa_handlers.msa_preprocessor import MSAPreprocessor

if __name__ == "__main__":
    # Get command line arguments and prepare directory
    cmd_liner = CmdHandler()
    msa = MSAPreprocessor(setuper=cmd_liner)
    msa.simple_preprocess_msa()
    print("#" * 80)
    print(f"     MSA for projection is ready in {cmd_liner.exp_dir}")
    print("#" * 80)
