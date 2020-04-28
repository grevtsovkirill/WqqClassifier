import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from samples import *

def data_load(in_list, do_clean=False):
    df = {}
    if do_clean:
        var_list=sel_vars()

    for s in samples:
        print(s,'  ',samples[s]['filename'])
        df[s] = pd.read_csv(BASE+samples[s]['filename'])
        if do_clean:
            df[s] = df[s][var_list]
    return df

def main():
    print("load data")
    dfs=data_load(samples)
     
if __name__ == "__main__":
    main() 
