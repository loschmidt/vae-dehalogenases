__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/06/07 11:30:00"

from enum import Enum


class Helper(Enum):
    DESCRIPTION = "VAE for ancestral reconstruction - parameters for the model"
    # Directory structure options
    EXP_DIR = "Experimental directory to gather the experiments in one directory."
    EXPERIMENT = "Setup output subdirectory in experimental directory for given experiment"
    # Model setup options
    MODEL_NAME = "Name your model to distinguish different setups while training"
    C = "Set parameter for training loss = (x - f(x)) - (1/C * KL(qZ, pZ)). Reconstruction parameter " \
        "and normalization parameter. The bigger C is more accurate the reconstruction will be. Default value is 2.0"
    K = "Cross validation iterations setup. Default is 5"
    subsets = "Train set should be split into many subsets, default 5"
    LAYERS = "List determining count of hidden layers and neurons within. Default 100. The dimensionality of latent " \
             "space is se over dimensionality argument. Example 2 3 "
    DIMS = "Latent space dimensionality. Default value 2"
    #MSA options
    REF = "The key of reference sequence for MSA processing; For instance P59336_S14"
    STATS = "Printing statistics of msa processing"
    ALIGN = "For highlighting. Align with reference sequence and then highlight in latent space. Sequences are passed " \
            "through highlight_files param in file"
    # EnzymeMiner scraper options
    PRESERVE = "Alternative filtering of MSA. Cooperate with EnzymeMiner, keep cat. residues. Use --ec_num param to " \
               "setup mine reference sequences."
    EC_NUM = "EC number for EnzymeMiner. Will pick up sequences from table and select with the most catalytic " \
             "residues to further processing."
    # Mutagenesis options
    MUT_POINTS = "Points of mutation. Default 1"
    MUT_SAMPLES = "Number of generated mutants, for mutagenesis analysis, default value is 100"
    ANCESTORS = "Number of generated mutants for straight ancestral strategy, default value is 100"
    # Pipelines options
    NO_SCORE = "Default. Loschmidt Labs pipeline for processing MSA."
    PAPER_LINE = "Original paper pipeline. Exclusive use score_filter and preserve_catalytics."
    # Highlight options
    HIGH_FILE = "Files with sequences to be highlighted. Array of files. Should be as the last param in case of usage"
    HIGH_SEQ = "Highlight sequences in dataset. Keys of MSA input are expected values"
    HIGH_DIM = "Inspect latent dimensions to see if they collapsed"
    HIGH_FOCUS = "Generate focus plot to the area with aligned ancestors"
    # Input MSA file
    MSA_FILE = "Setup input file. Recognize automatically .fasta or .txt for stockholm file format."
    # Clustal path option
    CLUSTAL = "Setup path to the clustal omega binary. Default is mine ./bin/clustao"
    # Robustness parameters
    ROB_TRAIN = "Option to run just one train fold validation for robustness purposes.\n It is used with robustness " \
                "class which inthis mode run batch scripts in cluster."
    ROB_MEA = "Option for robustness class. Run with this option in case you want measure \n the deviations from " \
              "referent model with name VAE_rob_0"
    LOG_DELIMETER = "="*80
    MODEL_FOLD = "_fold_0"

    MSA_NOT_CLUSTER = "Setup in the case you do not wish to cluster sequences via 90% identity"
    CONV = "Set this up to use encoder and decoder build up from from 2D convolution layers"
    CVAE = "Conditional variational autoencoder variant. The soluprot file should be provided in option solubility_file"
    SOL_FILE = "File with soluprot predictions for sequences in the dataset (proposed by enzymeMiner tool) "

class VaePaths(Enum):
    # Experiment directory paths
    RESULTS = "./../results/"
    MODEL_DIR = "model"
    PICKLE_DIR = "pickles"
    HIGHLIGHT_DIR = "highlight"
    STATS_DATA_SOURCE = "./../datasets/Statistics/data_source/"
    STATS_MAPPING_SOURCE = STATS_DATA_SOURCE + "/mapping_values/"
    FREQUENCIES_STATS = "Latent_space_stats"
    TREE_EVALUATION_DIR = "Tree_evaluation"
    MAPPING_DIR = "Mapping_to_latent_space"
    RECONSTRUCTOR = "Reconstruction/"
    BENCHMARK = "Benchmark/"
    BENCHMARK_DATA = BENCHMARK + "data/"
    BENCHMARK_STORE_FILE = BENCHMARK_DATA + "benchmark_data.pkl"
    # Files names
    MODEL_PARAMs_FILE = "ModelsParameters.txt"
    TRAIN_MSA_FILE = "training_alignment.pkl"


class ScriptNames(Enum):
    TRAIN = "train.py"
    MSA_PROCESS = "msa_preprocessor.py"
    VALIDATION = "validation_train.py"
    BENCH = "benchmark.py"


class SolubilitySetting(Enum):
    """ Setting for conditional vae and soluprot setting for bins division """
    SOLUBILITY_BINS = 3
    SOL_BIN_LOW = 0
    SOL_BIN_MEDIUM = 1
    SOL_BIN_HIGH = 2
