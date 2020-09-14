__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/27 11:30:00"

import argparse
import subprocess as sp   ## command line handling
import os

class StructChecker:
    def __init__(self):
        ## Setup all parameters
        self.pfam_id, self.args = self.get_parser()
        if self.args.ref is not None:
            self.ref_seq = True
            self.ref_n = self.args.ref
        else:
            self.ref_seq = False
            self.ref_n = ""
        self.keep_gaps = bool(self.args.keep_gaps)
        self.stats = self.args.stats
        self.epochs = self.args.num_epoch
        self.decay = self.args.weight_decay
        self.K = self.args.K # cross validation counts
        ## MSA processing handling
        self.preserve_catalytic = self.args.preserve_catalytic
        self.ec = self.args.ec_num
        self.filter_score = self.args.no_score_filter
        self.paper_pipe = self.args.paper_pipeline
        self.align = self.args.align
        self.mut_points = self.args.mut_points
        self.mutant_samples = self.args.mutant_samples
        self.focus = self.args.focus
        self.dimensionality = self.args.dimensionality

        self.highlight_files = self.args.highlight_files
        self.highlight_seqs = self.args.highlight_seqs

        ## Setup enviroment variable
        os.environ['PIP_CAT'] = str(self.preserve_catalytic)
        os.environ['PIP_PAPER'] = str(self.paper_pipe)
        os.environ['PIP_SCORE'] = str(self.filter_score)

        ## Directory structure variables
        self.res_root_fld = "./results/"
        self.run_root_dir = self.res_root_fld + self.pfam_id + "/"
        self.MSA_fld = self.run_root_dir + "MSA/"

        self.flds_arr = [self.res_root_fld, self.run_root_dir, self.MSA_fld]

    def add_to_dir(self, folderName):
        self.flds_arr.append(folderName)

    def remove_from_dir(self, folderName):
        self.flds_arr.remove(folderName)

    def set_msa_file(self, file_name):
        self.msa_file = file_name

    def get_parser(self):
        parser = argparse.ArgumentParser(description='Parameters for training the model')
        parser.add_argument("--Pfam_id", help="the ID of Pfam; e.g. PF00041, will create subdir for script data")
        parser.add_argument("--RP", help="RP specifier of given Pfam_id family, e.g. RP15, default value is full")
        parser.add_argument("--ref", help="the reference sequence; e.g. TENA_HUMAN/804-884")
        parser.add_argument("--keep_gaps", help="Setup in case you want to keep all gaps in sequences. Default False", default=False)
        parser.add_argument("--stats", action='store_true', help="Printing statistics of msa processing", default=False)
        parser.add_argument('--num_epoch', type=int, default=10000)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--output_dir', type=str, default=None, help="Option for setup output directory")
        parser.add_argument('--K', type=int, default=5, help="Cross validation iterations setup. Default is 5")
        parser.add_argument('--mut_points', type=int, default=1, help="Points of mutation. Default 1")
        parser.add_argument('--mutant_samples', type=int, default=5, help="Count of mutants will be generated for each ancestor. Defa")
        parser.add_argument('--preserve_catalytic', action='store_true', default=False, help="Alternative filtering of MSA. Cooperate with EnzymeMiner,"
                                                                                              " keep cat. residues. Use --ec_num param to setup mine reference sequences.")
        parser.add_argument('--ec_num', type=str, default="3.8.1.5",
                            help="EC number for EnzymeMiner. Will pick up sequences from table and select with the most "
                                 "catalytic residues to further processing.")
        parser.add_argument('--no_score_filter', action='store_false', default=True, help="Default. Loschmidt Labs pipeline for processing MSA.")
        parser.add_argument('--paper_pipeline', action='store_true', default=False,
                            help="Original paper pipeline. Exclusive use score_filter and preserve_catalytics.")
        parser.add_argument('--align', action='store_true', default=False,
                            help="For highlighting. Align with reference sequence and then highlight in latent space. Sequences are passed through highlight_files param in file")
        parser.add_argument('--highlight_files', type=str, default=None, help="Files with sequences to be highlighted. Array of files. Should be as"
                                                                              " the last param in case of usage")
        parser.add_argument('--highlight_seqs', type=str, default=None, help="Highlight sequences in dataset")
        parser.add_argument('--focus', action='store_true', default=False,
                            help="Generate focus plot")
        parser.add_argument('--dimensionality', type=int, default=2, help="Latent space dimensionality. Default value 2")
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
        """Setups subfolder of current rp group. Sets model, pickles"""
        self.rp = "full"
        if self.args.RP is not None:
            self.rp = self.args.RP
        self.rp_dir = self.run_root_dir + (self.rp if self.args.output_dir is None else self.args.output_dir.replace('/', '-'))
        self.VAE_model_dir = self.rp_dir + "/" + "model"
        self.pickles_fld = self.rp_dir + "/" + "pickles"
        self.high_fld = self.rp_dir + '/highlight'
        self.add_to_dir(self.rp_dir)
        self.add_to_dir(self.VAE_model_dir)
        self.add_to_dir(self.pickles_fld)
        self.add_to_dir(self.high_fld)
        print("The run will be continue with RP \"{0}\" and data will be generated to {1}"
              " (for different rp rerun script with --RP [e.g. rp75] paramater)".format(self.rp, self.rp_dir))

if __name__ == '__main__':

    '''Pipeline of our VAE data preparation, training, executing'''
    tar_dir = StructChecker()
    tar_dir.setup_struct()

    ## Our modules imports
    from download_MSA import Downloader
    from pipeline_importer import MSA
    from train import Train
    from analyzer import VAEHandler, Highlighter

    down_MSA = Downloader(tar_dir)
    msa = MSA(tar_dir)
    msa.proc_msa()
    Train(tar_dir, msa=msa).train()
    ## Create latent space
    VAEHandler(setuper=tar_dir).latent_space()
    ## Highlight
    highlighter = Highlighter(tar_dir)
    if tar_dir.highlight_files is not None:
        files = tar_dir.highlight_files.split()
        wait_high = True if len(files) == 1 and tar_dir.highlight_seqs is not None else False
        for f in files:
            highlighter.highlight_file(file_name=f, wait=wait_high)
    if tar_dir.highlight_seqs is not None:
        names = tar_dir.highlight_seqs.split()
        for n in names:
            highlighter.highlight_name(name=n)
