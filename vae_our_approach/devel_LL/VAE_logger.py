__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/01/06 14:33:00"

text_keeper, text_len = "", 0


# python3 msa_preprocessor.py --exp_dir simple --ref P59336_S14 --experiment refactor --in_file results/PF00561/MSA/identified_targets_msa.fa
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

