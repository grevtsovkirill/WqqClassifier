from samples import *
import matplotlib.pyplot as plt
import numpy as np  
scale_to_GeV=0.001
binning = {"DRll01": np.linspace(-2, 6, 24),
           "max_eta": np.linspace(0, 2.5, 26),
           "Njets": np.linspace(0, 10, 10),
           "mjj": np.linspace(0, 150, 150),
          }


def plot_var(df_bkg,lab_list,var,do_stack=True,sel_val=0,GeV=1):
    stack_var=[]
    stack_var_w=[]
    stack_var_leg=[]
    stack_var_yields=[]
    stack_var_s=[]
    stack_var_col=[]
    outname=''
    for i in lab_list:
        if sel_val==0:
            stack_var.append(df_bkg[i][var].loc[df_bkg[i].region==0]*GeV)
            stack_var_w.append(df_bkg[i].weight_tot.loc[df_bkg[i].region==0])
            stack_var_yields.append(df_bkg[i].weight_tot.loc[df_bkg[i].region==0].sum())
            yield_val = '{0:.2f}'.format(df_bkg[i].weight_tot.loc[df_bkg[i].region==0].sum())
        else:
            outname=str(sel_val)
            df_bkg[i] = df_bkg[i].loc[df_bkg[i].region==0]
            df_bkg[i] = df_bkg[i].loc[df_bkg[i].score>sel_val]
            stack_var.append(df_bkg[i][var]*GeV)
            stack_var_w.append(df_bkg[i].weight_tot)
            stack_var_yields.append(df_bkg[i].weight_tot.sum())
            yield_val = '{0:.2f}'.format(df_bkg[i].weight_tot.sum())
            
        
        #print(i," ", yield_val)
        stack_var_s.append(i)
        stack_var_leg.append(samples[i]['group']+" "+yield_val)
        stack_var_col.append(samples[i]['color'])

    if do_stack:
        plt.figure("stack") 
        plt.hist( stack_var, binning[var], histtype='step',
                  weights=stack_var_w,
                  label=stack_var_leg,
                  color = stack_var_col,
                  stacked=True, 
                  fill=True,
                  log=True,
                  linewidth=2,
                  alpha=0.8
        )
        plt.xlabel(var,fontsize=12)
        plt.ylim(1e-3, 1e6)
        plt.ylabel('# Events',fontsize=12) 
        plt.legend()
        plt.savefig('Outputs/stack/'+var+outname+'.png') #, transparent=True)
        plt.close("stack")
        with open("Outputs/stack/yields"+outname+".txt", "w") as f:
            f.write("Samples:{}\n".format(stack_var_s))
            f.write("Yields:{}\n".format(stack_var_yields))
            f.write("S/B:{}\n".format(stack_var_yields[0]/stack_var_yields[1]))
            f.write("S/sqrtB:{}\n".format(stack_var_yields[0]/np.sqrt(stack_var_yields[1])))
    else:
        plt.figure("norm") 
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
        plt.savefig('Outputs/norm/'+var+'.png', transparent=True)
        plt.close("norm")
