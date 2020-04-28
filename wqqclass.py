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

def plot_var(df_bkg,lab_list,var,do_stack=True,GeV=1):
    stack_var=[]
    stack_var_w=[]
    stack_var_leg=[]
    stack_var_col=[]
    for i in lab_list:
        stack_var.append(df_bkg[i][var].loc[df_bkg[i].region==0]*GeV)
        stack_var_w.append(df_bkg[i].weight_tot.loc[df_bkg[i].region==0])
        stack_var_leg.append(samples[i]['group'])
        stack_var_col.append(samples[i]['color'])

    if do_stack:
        plt.figure(1) 
        plt.hist( stack_var, binning[var], histtype='step',
                  weights=stack_var_w,
                  label=stack_var_leg,
                  color = stack_var_col,
                  stacked=True, 
                  fill=True,
                  linewidth=2, alpha=0.8)
        plt.xlabel(var,fontsize=12)
        plt.ylabel('# Events',fontsize=12) 
        plt.legend()
        plt.savefig('Plots/stack/'+var+'.png', transparent=True)
    else:
        plt.figure(2) 
        plt.hist( stack_var, binning[var], histtype='step',
                  weights=stack_var_w,
                  label=stack_var_leg,
                  color = stack_var_col,
                  density=1,
                  stacked=False, 
                  fill=False, 
                  linewidth=2, alpha=0.8)
        plt.xlabel(var,fontsize=12)
        plt.ylabel('# Events',fontsize=12) 
        plt.legend()
        plt.savefig('Plots/norm/'+var+'.png', transparent=True)
    
    
def main():
    print("load data")
    dfs=data_load(samples)
    #print(dfs['ttW'].columns)
    plot_var(dfs,['ttW','ttZmumu','ttbar'],'Njets')
    plot_var(dfs,['ttW','ttbar'],'Njets',False)
     
if __name__ == "__main__":
    main() 
