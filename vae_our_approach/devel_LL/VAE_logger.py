__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/01/06 14:33:00"

from vae_our_approach.devel_LL.metaclasses import Singleton

# python3 msa_filter_scorer.py --exp_dir simple --experiment refactor --in_file results/PF00561/MSA/identified_targets_msa.fa
class Logger(Singleton):
    """
    Logger class for all classes.
    """
