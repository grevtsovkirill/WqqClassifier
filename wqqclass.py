import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from samples import *

import argparse
parser = argparse.ArgumentParser(description='Prepare classifier')
parser.add_argument('-t','--type', required=True, type=str, choices=['plot', 'train','apply'], help='Choose processing type: explore variable [plot], train the model [train], or apply existing model [apply] ')
parser.add_argument('-s','--samples', nargs='+', default=['ttW','ttbar'], help='Choose list of samples to run over ')

args = parser.parse_args()

process_type = vars(args)["type"]
sample_list = vars(args)["samples"]

scale_to_GeV=0.001
binning = {"DRll01": np.linspace(-2, 6, 24),
           "max_eta": np.linspace(0, 2.5, 26),
           "Njets": np.linspace(0, 10, 10),
          }
def data_load(in_list, do_clean=False):
    df = {}
    if do_clean:
        var_list=sel_vars()

    for s in in_list:
        if s in samples:
            print(s,'  ',samples[s]['filename'])
            df[s] = pd.read_csv(BASE+samples[s]['filename'])
            if do_clean:
                df[s] = df[s][var_list]
        else:
            print(s,' is not in available sample list')
            break
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
    dfs=data_load(sample_list)
    #print(dfs['ttW'].columns)
    if process_type =='plot':
        plot_var(dfs,['ttW','ttZmumu','ttbar'],'Njets')
        plot_var(dfs,['ttW','ttbar'],'Njets',False)

    elif process_type == 'train':
        if dfs:
            print("prepare for training")

        
     
if __name__ == "__main__":
    main() 
