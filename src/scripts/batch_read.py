import re
from copy import deepcopy
from dataclasses import dataclass, InitVar, field
from easydict import EasyDict
from functools import reduce, partial,wraps
from icecream import ic
from matplotlib import pyplot as plt
from os import path
import argparse
import datetime,pytz
import dill
import json
import numpy as np
import os
import sys
import time
import logging
logging.basicConfig(level=getattr(logging, "INFO" if "LOG_LEVEL" not in os.environ else os.environ["LOG_LEVEL"]))
epochs = [ 1, 5, 10, 20, 30]
duplicants = [1,2,3,4,5]

exp_command_0_dict= {
    "vanilla":"echo vanilla",
    "diagnoal_double":"echo diagnoal_double",
    "concat":"set -xg DIAGONAL_DOUBLE_VARIANT_CONCAT concat",
    "sum":"set -xg DIAGONAL_DOUBLE_VARIANT_SUM sum",
}

exp_command_1_dict= {
    "vanilla":"diagonal",
    "diagnoal_double":"diagnoal_double",
    "concat":"diagnoal_double",
    "sum":"diagnoal_double",
}
commands = []
result_dict = {}
for exp in exp_command_0_dict.keys():
    for epoch in epochs:
        for duplicant in duplicants:
            exp_str = f"2_by_2_mult_double_10k_py_{exp}_exp{duplicant}_epochs{epoch}"
            data_str = f"2_by_2_mult_double_10k"
            
            try:
                with open(f"generation_logs/{exp_str}/gpt2/coupled.txt","r") as f:
                    lines = f.readlines()
                    lines = [line for line in lines if "Accuracy" in line]
                    
                    # Val. PPL: 2.0859139005884617; Accuracy: 0.0; Token Accuracy: 0.7200000286102295.
                    line=lines[-1]
                    # use regex to extract the accuracy in line
                    accuracy = re.findall("Accuracy: (.*?);",line)[0]
                    key = f"{exp}_exp{duplicant}_epochs{epoch}_coupled_accuracy"
                    result_dict[key] = accuracy
            except:
                logging.warning(f"generation_logs/{exp_str}/gpt2/coupled.txt not found")
print(result_dict)
