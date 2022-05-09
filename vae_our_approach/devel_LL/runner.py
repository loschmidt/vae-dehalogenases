__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/21 15:00:00"
__description__ = " Additional layer over project parser to simplify project setup "

import sys
import subprocess as sp
import json

runner_json = "model_configurations/runner-hts.json"

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

print("\n\t\tRunning with {} configuration file!".format(runner_json))

with open(runner_json, "r") as json_file:
    run_setup = json.load(json_file)

run_string = "python3 {} ".format(sys.argv[1])
if len(sys.argv) > 2:
    run_string += "{} ".format(sys.argv[2])  # in case of statistics mode

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
