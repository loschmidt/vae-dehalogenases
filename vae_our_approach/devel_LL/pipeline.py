__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/27 11:30:00"

import argparse
import subprocess as sp   ## command line handling

class StructChecker:
    def __init__(self):
        self.pfam_id, args = self.get_parser()

        self.res_root_fld = "./results/"
        self.run_name = self.res_root_fld + self.pfam_id + "/"
        self.MSA_fld = self.run_name + "MSA/"
        self.VAE_model_dir = self.run_name + "model"
        self.flds_arr = [self.res_root_fld, self.run_name, self.MSA_fld, self.VAE_model_dir]

    def add_to_dir(self, folderName):
        self.flds_arr.append(folderName)

    def remove_from_dir(self, folderName):
        self.flds_arr.remove(folderName)

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Parameters for training the model')
        parser.add_argument("--Pfam_id", help="the ID of Pfam; e.g. PF00041, will create subdir for script data")
        args = parser.parse_args()
        if args.Pfam_id is None:
            print("Error: Pfam_id parameter is missing!! Please run {0} --Pfam_id [Pfam ID]".format(__file__))
            exit(1)
        return args.Pfam_id, args

    def setup_struct(self):
        for folder in self.flds_arr:
            sp.run("mkdir -p {0}".format(folder), shell=True)
        print("Directory structure of {0} was successfully prepared".format(self.run_name))

if __name__ == '__main__':
    ## Our modules imports
    from download_MSA import Downloader

    '''Pipeline of our VAE data preparation, training, executing'''
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    down_MSA = Downloader(tar_dir)