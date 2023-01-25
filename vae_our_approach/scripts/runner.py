__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/21 15:00:00"
__description__ = " Additional layer over project parser to simplify project setup "

import os
import sys
import subprocess as sp
import json

runner_json = "model_configurations/runner-hts.json"

if len(sys.argv) < 4:
    print("\tYou did not specified proper cmd line input for script")
    print("\tPlease run as follow: "
          "\n\t\tpython3 runner.py [script to run] --json [path_to_config]")
    exit(1)

if sys.argv[1] in ["--help", "-h"]:
    print("")
    print("  Help of VAE ancestral reconstruction protocol:")
    print("\nThe run configuration file can be found in directory model_configuration/")
    print("\nrun .... for ....")
    print("\tpython3 runner.py msa_handlers/msa_preprocessor.py   for   MSA preprocessing")
    print("\tpython3 runner.py msa_handlers/train.py              for   Model training")
    print("\tpython3 runner.py run_task.py --[option]             for   Model further analysis")
    print("\nrun_task.py options:\n")
    sp.run("python3 run_task.py --help", shell=True)
    exit(0)

# Get param order for config
config_i = 2

# Run task on trained model
run_string = "python3 {} ".format(sys.argv[1])  # add script name to run
if sys.argv[1] == "run_task.py":
    run_string += "{} ".format(sys.argv[2])  # in case of statistics mode
    config_i += 1

if sys.argv[config_i] in ["--json", "--config"]:
    runner_json = sys.argv[config_i+1]
else:
    print("\tUse --json or --config options to pass configuration file!!!")
    exit(1)

print("\n\t\tRunning with {} configuration file!".format(runner_json))

with open(runner_json, "r") as json_file:
    run_setup = json.load(json_file)

# set environment variable to hold configuration file
os.environ["VAE_CONF"] = runner_json

if len(sys.argv) <= config_i + 2:
    pass
# we are running ensemble training, and we are passing ensemble number of model to be trained
else:
    if sys.argv[config_i+2] == "--ensemble_num":
        run_string += f"{sys.argv[config_i+2]} {sys.argv[config_i+3]} "

on_found = False
on_experiment = {}
for exp in run_setup["experiments"]:
    if exp["status"] == "on":
        if on_found:
            print(
                "  More experiments ({}, {}) status are set on. Please select desire one!".format(
                    on_experiment["experiment"]
                    , exp["experiment"]))
            exit()
        on_found = True
        on_experiment = exp

if len(on_experiment.keys()) == 0:
    print("   Please set one of configurations on!")
    exit(0)

print("\t\tSelected configuration is {}\n".format(on_experiment["experiment"]))

for param in run_setup["core_params"]:
    run_string += param + " "
for var_param in run_setup["variable_params"]:
    param_name = var_param.split(" ")[0][2:]
    run_string += var_param.format(on_experiment[param_name]) + " "
for flag in on_experiment["flags"]:
    run_string += "--{} ".format(flag)

print("  Running this command:\n      ", run_string)
sp.run(run_string, shell=True)
