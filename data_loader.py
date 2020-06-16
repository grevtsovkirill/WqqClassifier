import pandas as pd
import numpy as np
from samples import *
import json


def sel_vars(list_name="varlist.json"):
    with open(list_name) as vardict:
        variablelist = json.load(vardict)[:]

    print(variablelist)
    return variablelist

def data_load(in_list, do_clean):
    df = {}
    if do_clean:
        var_list=sel_vars()

    for s in in_list:
        if s in samples:
            print(s,'  ',samples[s]['filename'])
            flist = []
            for i in samples[s]['filename']:
                print(i)
                dftmp = pd.read_csv(BASE+i, index_col=None, header=0)
                flist.append(dftmp)

            df[s] = pd.concat(flist, axis=0, ignore_index=True)
            #df[s] = pd.read_csv(BASE+samples[s]['filename'])
            #df[s] = df[s].loc[(df[s].mjj>60) & (df[s].mjj<100)]
            #df[s] = df[s].loc[(df[s].mjj<150000)]
            df[s] = df[s].loc[(df[s].Njets==4) | (df[s].Njets==5)]
            if do_clean:
                #df[s] = df[s].loc[df[s].region==0]
                df[s] = df[s][var_list]
        else:
            print(s,' is not in available sample list')
            break
    return df

    
