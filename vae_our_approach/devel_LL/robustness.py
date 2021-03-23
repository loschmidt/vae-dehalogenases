__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2021/03/23 10:26:00"

import os
import pickle

from pipeline import StructChecker

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
                It samples straight ancestors from first reference model. These sequences lie on the line
                in reference model, but where are they located in other models?
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
        pass

if __name__ == '__main__':
    tar_dir = StructChecker()
    tar_dir.setup_struct()
    Robustness(setuper=tar_dir)
