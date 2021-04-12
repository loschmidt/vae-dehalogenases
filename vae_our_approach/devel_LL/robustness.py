__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/03/23 10:26:00"

import os
import pickle

from pipeline import StructChecker
from analyzer import VAEHandler, Highlighter
from msa_filter_scorer import MSAFilterCutOff as Convertor
from math import sqrt

class Robustness:
    """
        The purpose of this class is to execute robustness measurement of
        VAE ancestral reconstruction approach.
        The class runs in two modes:
            i) --robustness_train option:
                No models are trained. So train models in cluster via batch scripts execution with
                random weights initialization.
                Prepare configuration file for measure part and write there all important stuffs.
            ii) --robustness_measure option:
                This mode is run after training of all models have been done. User has to run it on his own
                it is not automated.
                It samples straight ancestors from REFERENCE model (this model has to be named reference).
                These sequences lie on the line in reference model, but where are they located in other models?
                These sequences are then embedded into the latent space of rest set of models and the embedding
                deviation from line is done.
    """

    def __init__(self, setuper):
        self.setuper = setuper
        self.out_dir = setuper.high_fld + '/'
        self.pickle = setuper.pickles_fld + '/'
        self.name = 'robustness'
        self.filename = self.pickle + '{}_models.pkl'.format(self.name)

        TRAIN = 0
        MEASURE = 1

        self.mode = TRAIN if self.setuper.robustness_train else 0
        self.mode = MEASURE if self.setuper.robustness_measure else 0

        if self.mode == TRAIN:
            self.robustness_train()
        else:
            self.high = Highlighter(setuper)
            self.robustness_measure()

    def robustness_train(self):
        """ Run special batch scripts with desire parameters. Store info to file """
        # store the model setup to file
        model_cnt = 10
        model_names = []
        model_name = 'VAE_rob_'
        for i in range(model_cnt):
            model_names.append(model_name + str(i))
        # Store models names
        with open(self.filename, 'wb') as file_handle:
            pickle.dump(model_names, file_handle)

        print("Robustness message : initialization of {} models training".format(model_cnt))

        for name in model_names:
            cmd = 'qsub -v name="{}" ./run_scripts/robustness_train.sh'.format(name)
            print("Robustness message : running cmd", cmd)
            os.system(cmd)

    def robustness_measure(self):
        """ Measure robustness of random weights initialization """
        models = self._get_successful_models()
        convector = Convertor(self.setuper)
        ancestors = self._get_original_straight_ancestors()
        binary, weights, keys = convector.prepare_aligned_msa_for_Vae(ancestors)
        # Prepare query sequence
        with open(self.pickle + "reference_seq.pkl", 'rb') as file_handle:
            ref_dict = pickle.load(file_handle)
        query_bin, query_w, query_k = convector.prepare_aligned_msa_for_Vae(ref_dict)

        for model in models:
            # Get positions of straight ancestors in model and compute derivation
            vae = VAEHandler(self.setuper, model_name=model)
            query_pos, _ = vae.propagate_through_VAE(query_bin, query_w, query_k)
            data, _ = vae.propagate_through_VAE(binary, weights, keys)
            mean, maxDev = Robustness.compute_deviation(query_pos, data)
            self.high.highlight_line_deviation(query_pos, data, mean, maxDev, file_name=model + '_robustPlot')
            del vae

    @staticmethod
    def compute_deviation(query, points):
        """
            Measure how far points of ancestors are from straight line
            toward center from query position.
            To compute distance of point from line use formula for computation distance
            from line define by two points in our case by center (0, 0) and query position (query[0], query[1])

            distance(p1,p2, (x0, y0)) =  |(x2 - x1)(y1 - y0) - (x1 - x0)(y2 - y1)| / sqrt((x2 - x1)^2 + (y2 - y1)^2
        """
        x1, y1 = (0, 0)
        x2, y2 = tuple(query[0])
        denominator = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        deviations = []
        for x0, y0 in points:
            v = abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)) / denominator
            deviations.append(v)
        av_dev = sum(deviations) / len(deviations)
        return av_dev, max(deviations)

    def _get_original_straight_ancestors(self):
        """ Loading sequences generated during straight ancestral reconstruction with reference model """
        filename = self.out_dir + "referencestraight_ancestors.fasta"
        if not os.path.isfile(filename):
            print("Robustness message : Not find reference straight ancestors in file", filename)
            exit(0)
        from msa_prepar import MSA
        self.setuper.set_msa_file("tmp") # Secure that msa file attribute exists
        m = MSA(self.setuper, processMSA=False)
        ancs = m.load_msa(filename)
        return ancs

    def _get_successful_models(self, filename="ModelsRobustnessLosses.txt"):
        """ Parsing file with names of models and select that which were successful"""
        filename = self.setuper.pickles_fld + '/' + filename
        model_names = []
        model_losses = []
        with open(filename, "r") as file:
            # Format Model name, NAME, loss, LOSS
            for l in file:
                data = l.split(sep=",")
                model_names.append(data[1] + ".model")
                model_losses.append(data[3])
        print("Robustness message : filtering models by loss success, cnt of models {}".format(len(model_names)))
        # Model with loss bigger than 3000 is excluded
        final_models = []
        for loss, model in zip(model_losses, model_names):
            if float(loss) < 500:
                final_models.append(model)
        print("Robustness message : {} models left after filtering phase".format(len(final_models)))
        return final_models


if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    Robustness(setuper=tar_dir)
