# All import statements
import sys
print(sys.version)

import numpy as np
import pandas as pd 
import json
import os

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Relu function
def relu(x):
    return np.max(0, x)        

all_data = []
with open("wdbc.dat") as fn:
    content = fn.readlines()
    for line in content:
        stripped_line = line.rstrip().split(",")
        all_data.append(float(stripped_line) if is_float(stripped_line) else i for i in stripped_line)


print(content) 