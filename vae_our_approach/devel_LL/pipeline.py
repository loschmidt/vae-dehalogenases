__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/27 11:30:00"

import argparse
import subprocess as sp   ## command line handling

class StructChecker:
    def __init__(self):
        self.pfam_id, self.args = self.get_parser()

        self.res_root_fld = "./results/"
        self.run_root_dir = self.res_root_fld + self.pfam_id + "/"
        self.MSA_fld = self.run_root_dir + "MSA/"

        self.flds_arr = [self.res_root_fld, self.run_root_dir, self.MSA_fld]

    def add_to_dir(self, folderName):
        self.flds_arr.append(folderName)

    def remove_from_dir(self, folderName):
        self.flds_arr.remove(folderName)

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Parameters for training the model')
        parser.add_argument("--Pfam_id", help="the ID of Pfam; e.g. PF00041, will create subdir for script data")
        parser.add_argument("--RP", help="RP specifier of given Pfam_id family, e.g. RP15, default value is full")
        args = parser.parse_args()
        if args.Pfam_id is None:
            print("Error: Pfam_id parameter is missing!! Please run {0} --Pfam_id [Pfam ID]".format(__file__))
            exit(1)
        return args.Pfam_id, args

    def setup_struct(self):
        self._setup_RP_run()
        for folder in self.flds_arr:
            sp.run("mkdir -p {0}".format(folder), shell=True)
        print("Directory structure of {0} was successfully prepared".format(self.run_root_dir))

    def _setup_RP_run(self):
        self.rp = "full"
        if self.args.RP is not None:
            self.rp = self.args.RP
        self.rp_dir = self.run_root_dir + "/" + self.rp
        self.VAE_model_dir = self.rp_dir + "/" + "model"
        self.add_to_dir(self.rp_dir)
        self.add_to_dir(self.VAE_model_dir)
        print("The run will be continue with RP \"{0}\" and data will be generated to {1}"
              " (for different rp rerun script with --RP [e.g. rp75] paramater)".format(self.rp, self.rp_dir))

if __name__ == '__main__':
    ## Our modules imports
    from download_MSA import Downloader

    '''Pipeline of our VAE data preparation, training, executing'''
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    down_MSA = Downloader(tar_dir)