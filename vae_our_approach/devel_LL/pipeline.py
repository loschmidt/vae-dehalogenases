__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/27 11:30:00"

import argparse
import subprocess as sp   ## command line handling

class StructChecker:
    def __init__(self):
        pfam_ID, args = self.get_parser()

        self.res_root_fld = "./PIPE_RES/"
        self.run_name = self.res_root_fld + pfam_ID + "/"
        self.MSA_fld = self.run_name + "MSA/"
        self.flds_arr = [self.res_root_fld, self.run_name, self.MSA_fld]

    def add_to_check(self, folderName):
        self.flds_arr.append(folderName)

    def do_not_check(self, folderName):
        self.flds_arr.remove(folderName)

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Parameters for training the model')
        parser.add_argument("--Pfam_id", help="the ID of Pfam; e.g. PF00041, will create subdir for script data")
        args = parser.parse_args()
        if args.Pfam_id is None:
            print("Error: Pfam_id parameter is missing!! Please run {0} --Pfam_id \"tip your Pfam ID\"".format(__file__))
            exit(1)
        return args.Pfam_id, args

    def check_struct(self):
        for folder in self.flds_arr:
            sp.run("mkdir -p {0}".format(folder), shell=True)

if __name__ == '__main__':
    '''Pipeline of our VAE data preparation, training, executing'''
