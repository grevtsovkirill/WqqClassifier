from samples import *
import matplotlib.pyplot as plt
import numpy as np  
scale_to_GeV=0.001
binning = {"drll01": [np.linspace(-2, 6, 24),1],
           "eta": [np.linspace(0, 2.5, 26),1],
           "Njets": [np.linspace(0, 10, 10),1],
           "mjj": [np.linspace(0, 150, 150),scale_to_GeV],
           "score": [np.linspace(0, 1, 20),1]
          }


def plot_var(df_bkg,lab_list,var,do_stack=True,sel_val=0):
    stack_var=[]
    stack_var_w=[]
    stack_var_leg=[]
    stack_var_yields=[]
    stack_var_s=[]
    stack_var_col=[]
    outname=''
    bins = []
    GeV = 1
    for i,j in binning.items():
        if i in var:
            bins = j[0]
            GeV = j[1]
            
    #print(bins,GeV)

    for i in lab_list:
        if sel_val==0:
            #print(df_bkg[i][var].loc[df_bkg[i].region==0].head())
            #stack_var.append(df_bkg[i][var].loc[df_bkg[i].region==0]*GeV)
            stack_var.append(df_bkg[i][var]*GeV)
            stack_var_w.append(df_bkg[i].weight_tot)
            stack_var_yields.append(df_bkg[i].weight_tot.sum())
            yield_val = '{0:.2f}'.format(df_bkg[i].weight_tot.sum())
            print( yield_val)
        else:
            outname=str(sel_val)
            #print(df_bkg[i][var].loc[df_bkg[i].region==0].head())
            #df_bkg[i] = df_bkg[i].loc[df_bkg[i].region==0]
            df_bkg[i] = df_bkg[i].loc[df_bkg[i].score>sel_val]
            stack_var.append(df_bkg[i][var]*GeV)
            stack_var_w.append(df_bkg[i].weight_tot)
            stack_var_yields.append(df_bkg[i].weight_tot.sum())
            yield_val = '{0:.2f}'.format(df_bkg[i].weight_tot.sum())
            #print(outname,yield_val)
        
        print("lab_list = ", i," yield_val= ", yield_val)
        stack_var_s.append(i)
        stack_var_leg.append(samples[i]['group']+" "+yield_val)
        stack_var_col.append(samples[i]['color'])
        #print("append leg, col")
    if do_stack:
        #print("do stack, stack_var=",stack_var)
        #print(", binning[var]=",binning[var])
        #print(", stack_var_leg",stack_var_leg)
        #print("stack_var_w",stack_var_w)
        plt.figure("stack") 
        plt.hist( stack_var, bins, histtype='step',
                  weights=stack_var_w,
                  label=stack_var_leg,
                  color = stack_var_col,
                  stacked=True, 
                  fill=True,
                  log=True,
                  linewidth=2,
                  alpha=0.8
        )
        #print("fill hist")
        plt.xlabel(var,fontsize=12)
        plt.ylim(1e-3, 1e6)
        plt.ylabel('# Events',fontsize=12) 
        plt.legend()
        #print("plot leg")
        plt.savefig('Outputs/stack/'+var+outname+'.png') #, transparent=True)
        plt.close("stack")
        #print("do yields")
        with open("Outputs/stack/yields"+outname+".txt", "w") as f:
            f.write("Samples:{}\n".format(stack_var_s))
            f.write("Yields:{}\n".format(stack_var_yields))
            f.write("S/B:{}\n".format(stack_var_yields[0]/stack_var_yields[1]))
            f.write("S/sqrtB:{}\n".format(stack_var_yields[0]/np.sqrt(stack_var_yields[1])))
    else:
        print("do norm")
        plt.figure("norm") 
        plt.hist( stack_var, bins, histtype='step',
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
