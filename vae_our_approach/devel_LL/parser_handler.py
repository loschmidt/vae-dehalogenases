__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2020/07/27 11:30:00"
__description__ = " Handling command line arguments and setting up and checking the structure of result directories." \
                  " Handles models names, loading of parameters from log files for given experiments and so on "

import argparse
import datetime
import os
import pickle
import subprocess as sp  # command line handling
import sys

from typing import Dict
from project_enums import Helper, VaePaths, ScriptNames
# from sequence_transformer import Transformer


class CmdHandler:
    def __init__(self):
        # Setup all parameters
        self.exp_dir, self.args = self.get_parser()

        self.msa_file = ""
        self.stats = self.args.stats
        self.epochs = self.args.num_epoch
        self.decay = self.args.weight_decay
        self.K = self.args.K  # cross validation counts
        self.C = self.args.C
        # MSA processing handling
        self.preserve_catalytic = self.args.preserve_catalytic
        self.ec = self.args.ec_num
        self.filter_score = self.args.no_score_filter
        self.paper_pipe = self.args.paper_pipeline
        self.align = self.args.align
        self.mut_points = self.args.mut_points
        self.focus = self.args.focus
        self.dimensionality = self.args.dimensionality

        self.highlight_files = self.args.highlight_files
        self.highlight_seqs = self.args.highlight_seqs
        self.highlight_instricts = self.args.highlight_dim

        self.in_file = self.args.in_file
        self.clustalo_path = self.args.clustalo_path

        self.robustness_train = self.args.robustness_train
        self.robustness_measure = self.args.robustness_measure

        self._handle_layers()

        self.model_name = self.args.model_name # + Helper.MODEL_FOLD.value

        ## Setup enviroment variable
        os.environ['PIP_CAT'] = str(self.preserve_catalytic)
        os.environ['PIP_PAPER'] = str(self.paper_pipe)
        os.environ['PIP_SCORE'] = str(self.filter_score)

        # Prepare project structure
        self.setup_struct()

        # Load ref sequence if not specified
        if self.args.query is not None:
            self.query_id = self.args.query
        else:
            query = self.load_reference_sequence()
            self.query_id = list(query.keys())[0]

    def get_model_to_load(self):
        """ Get name of the model and we are loading zero fold """
        return self.model_name + Helper.MODEL_FOLD.value + ".model"

    def set_msa_file(self, file_name):
        self.msa_file = file_name

    def get_parser(self):
        parser = argparse.ArgumentParser(description=Helper.DESCRIPTION.value)
        # Directory structure options
        parser.add_argument("--exp_dir", help=Helper.EXP_DIR.value, type=str, default="default")
        parser.add_argument('--experiment', type=str, default="default", help=Helper.EXPERIMENT.value)
        # MSA options
        parser.add_argument("--query", help=Helper.REF.value)
        parser.add_argument("--stats", action='store_true', help=Helper.STATS.value, default=False)
        parser.add_argument('--align', action='store_true', default=False, help=Helper.ALIGN.value)
        # Mutagenesis options
        parser.add_argument('--mut_points', type=int, default=1, help=Helper.MUT_POINTS.value)
        # EnzymeMiner scraper options
        parser.add_argument('--preserve_catalytic', action='store_true', default=False, help=Helper.PRESERVE.value)
        parser.add_argument('--ec_num', type=str, default="3.8.1.5", help=Helper.EC_NUM.value)
        # Pipeline options
        parser.add_argument('--no_score_filter', action='store_false', default=True, help=Helper.NO_SCORE.value)
        parser.add_argument('--paper_pipeline', action='store_true', default=False, help=Helper.PAPER_LINE.value)
        # Highlight options
        parser.add_argument('--highlight_files', type=str, default=None, help=Helper.HIGH_FILE.value)
        parser.add_argument('--highlight_seqs', type=str, default=None, help=Helper.HIGH_SEQ.value)
        parser.add_argument('--highlight_dim', action='store_true', default=False, help=Helper.HIGH_DIM.value)
        parser.add_argument('--focus', action='store_true', default=False, help=Helper.HIGH_FOCUS.value)
        # Input MSA file
        parser.add_argument('--in_file', type=str, default='', help=Helper.MSA_FILE.value)
        # Model setup options
        parser.add_argument('--model_name', type=str, default="model", help=Helper.MODEL_NAME.value)
        parser.add_argument('--C', type=float, default=2.0, help=Helper.C.value)
        parser.add_argument('--num_epoch', type=int, default=16500)
        parser.add_argument('--weight_decay', type=float, default=0.0)
        parser.add_argument('--K', type=int, default=5, help=Helper.K.value)
        parser.add_argument('--layers', nargs='+', type=int, default=100, help=Helper.LAYERS.value)
        parser.add_argument('--dimensionality', type=int, default=2, help=Helper.DIMS.value)
        # Clustal path option
        parser.add_argument('--clustalo_path', type=str, default='/storage/brno2/xkohou15/bin/clustalo',
                            help=Helper.CLUSTAL.value)
        # Robustness options
        parser.add_argument('--robustness_train', action='store_true', default=False, help=Helper.ROB_TRAIN.value)
        parser.add_argument('--robustness_measure', action='store_true', default=False, help=Helper.ROB_MEA.value)
        args, unknown = parser.parse_known_args()
        if ('--source_txt' not in unknown and len(unknown) > 0) and \
                ('--run_package_stats' not in unknown and len(unknown) > 0):
            print(' Parser error : unrecognized parameters', unknown)
            exit(1)
        return args.exp_dir, args

    def setup_struct(self):
        """ Setup experiment directory and log its setup. Also init singleton classes """
        self._setup_experimental_paths()
        self._log_run_setup_into_file()
        self._load_model_params()
        print("\n" + Helper.LOG_DELIMETER.value)
        print(" Parser Handler message: : running with parameters\n"
              "                         weight decay   : {}\n"
              "                         layers setup   : {}\n"
              "                         dimensionality : {}\n"
              "                         C parameter    : {}\n"
              "                         Epochs count   : {}\n"
              "                         Model name     : {}".format(self.decay, self.layers, self.dimensionality,
                                                                    self.C, self.epochs, self.model_name))
        print(Helper.LOG_DELIMETER.value)
        # Transformer(setuper=self)

    def load_reference_sequence(self) -> Dict[str, str]:
        """ If in passed command line arguments ref option is missing try to find query in folder """
        query_file = self.pickles_fld + "/reference_seq.pkl"
        if not os.path.exists(query_file):
            print(" Parser_handler ERROR: The query sequence is neither provided via --ref option nor \n"
                  "                       stored in {}".format(query_file))
        with open(query_file, 'rb') as file_handle:
            query_dict = pickle.load(file_handle)
        print(" Parser Handler message: The query {} loaded". format(list(query_dict.keys())[0]))
        return query_dict

    def _handle_layers(self):
        """ Method decodes --layers argument and creates its string representation"""
        # if sys.argv[0] != ScriptNames.TRAIN.value:
        #     self.layersString, self.layers = "", []
        #     return
        # Decode layer setup in form [100 | 100 50]
        s_layers = 'L'
        self.layers = []
        try:
            for layer in self.args.layers:
                self.layers.append(layer)
                s_layers += '_{}'.format(layer)
        except TypeError:
            self.layers = [self.args.layers]
            s_layers += '_{}'.format(self.args.layers)
        self.layersString = s_layers

    def _log_run_setup_into_file(self):
        """ Log run setup into the file to keep track of experiments """
        filename = self.high_fld + "/" + VaePaths.MODEL_PARAMs_FILE.value
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        append_write = 'w'  # make a new file if not
        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        # Prepare logs into file
        model_str = "weight decay {},layer {},Dim {},C {},epochs {}".format(self.decay, self.layersString,
                                                                            self.dimensionality, self.C, self.epochs)
        running_script = os.path.basename(sys.argv[0])
        log_str = running_script + ";" + timestamp + ";"
        if running_script in [ScriptNames.TRAIN.value, ScriptNames.MSA_PROCESS.value, ScriptNames.VALIDATION.value]:
            log_str += "Model name;{}; training with parameters;{}\n".format(self.model_name, model_str)
        else:
            log_str += "Model name;{}; run with script {}\n".format(self.model_name, running_script)
        hs = open(filename, append_write)
        hs.write(log_str)
        hs.close()

    def _setup_experimental_paths(self):
        """ Method setups directory structure and create all necessary directories """
        # Directory structure variables
        root_dir = VaePaths.RESULTS.value + self.exp_dir + "/"
        _dir = root_dir + self.args.experiment.replace('/', '-')
        self.VAE_model_dir = _dir + "/" + VaePaths.MODEL_DIR.value
        self.pickles_fld = _dir + "/" + VaePaths.PICKLE_DIR.value
        self.high_fld = _dir + "/" + VaePaths.HIGHLIGHT_DIR.value

        dir_list = [VaePaths.RESULTS.value, root_dir, _dir, self.VAE_model_dir, self.pickles_fld, self.high_fld]
        for folder in dir_list:
            sp.run("mkdir -p {0}".format(folder), shell=True)
        print(" StructChecker message : the output directory is", _dir)

    def _load_model_params(self):
        """
        Load parameters of model given by --model_name argument options.
        Do not so in the case of training phase
        """
        # Scripts creating or working without model do not need to load its parameters
        if os.path.basename(sys.argv[0]) in \
                [ScriptNames.TRAIN.value, ScriptNames.MSA_PROCESS.value, ScriptNames.VALIDATION.value]:
            return
        model_params_file = self.high_fld + "/" + VaePaths.MODEL_PARAMs_FILE.value
        with open(model_params_file, "r") as file_handle:
            lines = file_handle.readlines()
            for line in reversed(lines):
                split_line = line.split(";")
                if split_line[0] != ScriptNames.TRAIN.value:
                    continue
                model_name, params = split_line[3], split_line[5].split(",")
                if model_name != self.model_name:
                    continue
                return self._parse_model_params(params)
        print(" StructChecker error : The model {} not found in {} file!".format(self.model_name, model_params_file))
        exit(1)

    def _parse_model_params(self, param):
        """ Parse model parameters logged in ModelParams file essential for model creation """
        self.dimensionality = int(param[2].split()[1])
        layers_str = param[1].split()[1]
        layers = layers_str.split("_")[1:]

        self.layers = []
        for layer in layers:
            self.layers.append(int(layer))
        self.layersString = layers_str


if __name__ == '__main__':

    '''Pipeline of our VAE data preparation, training, executing'''
    tar_dir = CmdHandler()

    # Our modules imports
    from analyzer import Highlighter
    from VAE_accessor import VAEAccessor

    # Create latent space
    VAEAccessor(setuper=tar_dir,model_name=tar_dir.get_model_to_load()).latent_space()
    # Highlight
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
