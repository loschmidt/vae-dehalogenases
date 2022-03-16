__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/01/06 14:33:00"

from io import StringIO
import sys

text_keeper, text_len = "", 0


# $python3 benchmark.py --model_name bench --exp_dir simple --query DhaA_S19 --experiment dhlA_LinB_DhaA_S19 --in_file results/PF00561/MSA/dhlA_LinB.fa --stats --num_epoch 5000 --layers 291 --robustness_train
class Logger:
    """
    Logger class for all classes.
    """

    @staticmethod
    def print_for_update(msg: str, value: str):
        """ Store to global variable. Msg should include {} fields for updatable content"""
        global text_keeper, text_len
        text_keeper = msg
        text_len = len(text_keeper.format(value))
        print(text_keeper.format(value), end='', flush=True)

    @staticmethod
    def update_msg(value: str, new_line=False):
        """ Update already printed message, new_line set to True if you wish to end updating line """
        global text_keeper, text_len

        # Update line with correct text, fit the length due to overwriting output
        msg = text_keeper.format(value)
        while len(msg) < text_len:
            msg += " "
        text_len = len(msg)
        print("\r{}".format(msg), end='', flush=True)

        if new_line:
            print()


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout
