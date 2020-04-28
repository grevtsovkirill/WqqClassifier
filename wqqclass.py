import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from samples import *


scale_to_GeV=0.001
binning = {"DRll01": np.linspace(-2, 6, 24),
           "max_eta": np.linspace(0, 2.5, 26),
           "Njets": np.linspace(0, 10, 10),
          }
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

def plot_stack_var(df_bkg,lab_list,var,GeV):
    stack_var=[]
    stack_var_w=[]
    stack_var_leg=[]    
    for i in lab_list:
        stack_var.append(df_bkg[i][var].loc[df_bkg[i].region==0]*GeV)
        stack_var_w.append(df_bkg[i].weight_tot.loc[df_bkg[i].region==0])
        stack_var_leg.append(i)

    plt.hist( stack_var, binning[var], histtype='step',
         weights=stack_var_w,
         label=stack_var_leg,
         stacked=True, 
         fill=True, 
         linewidth=2, alpha=0.8)
    plt.xlabel(var,fontsize=12)
    plt.ylabel('# Events',fontsize=12) 
    plt.legend()
    plt.savefig('Plots/'+var+'.png', transparent=True)

    
def main():
    print("load data")
    dfs=data_load(samples)
    #print(dfs['ttW'].columns)
    plot_stack_var(dfs,['ttW','ttZmumu'],'Njets',1)
     
if __name__ == "__main__":
    main() 
