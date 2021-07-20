from numpy.random import exponential as exp
from numpy.random import random as rand
import numpy as np
import math
import pandas as pd

TEST_DATA_PATH = "/Users/isaacbevers/multilingual-aoa-prediction/Data/model-sets/train.txt"
import time

start = time.time()
hello = pd.read_csv(TEST_DATA_PATH, sep = '\n', header = None)
end = time.time()
print(end - start)

start = time.time()
with open(TEST_DATA_PATH, "r") as f:
    hello = f.readlines()
    # self.data = f.readlines()
end = time.time()
print(end - start)