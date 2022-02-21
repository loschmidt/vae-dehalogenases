__author__ = "Pavel Kohout <xkohou15@stud.fit.vutbr.cz>"
__date__ = "2022/02/21 15:00:00"
__description__ = " Additional layer over project parser to simplify project setup "

import sys
import subprocess as sp
import json

runner_json = "runner.json"

with open(runner_json, "r") as json_file:
    run_setup = json.load(json_file)

run_string = "python3 {} ".format(sys.argv[1])

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

for param in run_setup["core_params"]:
    run_string += param + " "
for var_param in run_setup["variable_params"]:
    param_name = var_param.split(" ")[0][2:]
    run_string += var_param.format(on_experiment[param_name]) + " "

# print("  Running this command:\n      ", run_string)
sp.run(run_string, shell=True)
