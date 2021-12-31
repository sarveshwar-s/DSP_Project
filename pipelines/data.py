import pandas as pd
import numpy as np
import glob

all_data = []
# for item in glob.glob("../data/daily_dataset/*.csv"):
#     file = pd.read_csv(item)

data = pd.read_csv("../data/daily_dataset/block_0.csv")

lcl_list = data["LCLid"].value_counts()
for item in lcl_list:
    data[data[item] == item]