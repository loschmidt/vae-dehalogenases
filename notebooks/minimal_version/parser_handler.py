__author__ = "Pavel Kohout <pavel.kohout@recetox.muni.cz>"
__date__ = "2024/08/12"
__description__ = " Importing the and parsing of configuration file and setup run environment"

import datetime
import json
import os


class RunSetup:
    def __init__(self, config_file_path=None):
        """
        Parse and prepare structure for model run
        :param config_file_path: custom configuration file

        use configuration value -1 to use default value of parameter
        """

        with open(config_file_path, 'r') as file:
            conf = json.load(file)

        # setup directory structure if needed
        directories = ["model", "mapping", "msa", "pickles", "results", "config", "logs", "conditional_data",
                       "benchmark"]
        for exp_dir in directories:
            attr_dir = os.path.join(conf['paths']["result"], exp_dir)
            os.makedirs(attr_dir, exist_ok=True)
            setattr(self, exp_dir, attr_dir)
        self.root_dir = conf['paths']["result"]
        # set the rest of the keys
        for k, v in conf.items():
            for attr, attr_val in v.items():
                if attr_val == -1:  # default values will be used
                    continue
                setattr(self, attr, attr_val)
        # copy config file to the results directory
        now = datetime.datetime.now()
        file_name = now.strftime("%Y-%m-%d_%H-%M.json")
        json.dump(conf, open(os.path.join(self.config, f"{file_name}"), 'w'))
        print(f"Configuration file stored in {os.path.join(self.config, file_name)}")

    def __getattr__(self, attr_name):
        """Set the default values if not specified in configuration file"""
        fallback_values = {
            "clustering": False,
            "dynamic_decay": True,
            "decay": 0.01,
            "lat_dim": 2,
            "layers": None,
            "epochs": None,
            "evo_query": self.query,
            "ancestors": 100,
            "batch_size": None,
            "fixed_sequences": [],
            "model": "vae",
            "encoder-decoder": "dense",
            "K": 1,
            "run_capacity_test": False,
            "weights": os.path.join(self.model, "vae_fold_0.model")
        }
        if attr_name in fallback_values:
            return fallback_values[attr_name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")