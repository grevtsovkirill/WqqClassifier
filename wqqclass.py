import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from samples import *
import data_loader as dl
import model as md
import plotter as pl
import helpers as hp

seed=8
np.random.seed(seed)



import argparse
parser = argparse.ArgumentParser(description='Prepare classifier')
parser.add_argument('-t','--type', required=True, type=str, choices=['plot', 'train','read','apply'], help='Choose processing type: explore variable [plot], train the model [train], load previously trained model to do plots [read] or apply existing model [apply] ')
parser.add_argument('-s','--samples', nargs='+', default=['ttW','ttbar'], help='Choose list of samples to run over ')
parser.add_argument('-c','--clean', default=False, help='Use selected list of variables ')

args = parser.parse_args()

process_type = vars(args)["type"]
sample_list = vars(args)["samples"]
doclean = vars(args)["clean"]

if process_type =='train' or process_type == 'read' or process_type == 'apply' :

    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    from sklearn.model_selection import train_test_split
    from keras.layers import Layer, Input, Dense, Dropout
    from keras.models import Sequential, load_model
    from keras.callbacks import EarlyStopping, ModelCheckpoint






def do_plot(dfs,sample_list):
    varl = ['drll01','Njets','max_eta','mjj']
    varl = ['max_eta']
    for i in varl:
        pl.plot_var(dfs,sample_list,i)
        pl.plot_var(dfs,sample_list,i,False)
    

def main():
    print("load data")

    dfs=dl.data_load(sample_list,doclean)
    print(dfs['ttW'].columns)
    if process_type =='plot':
        do_plot(dfs,sample_list)

    elif process_type == 'apply':
        print("apply mode")
        model = load_model('Outputs/training/model_nn_v0.h5')
        with open('Outputs/training/scaler.pickle', 'rb') as f:
            sc = pickle.load(f)

            #def model_create_feature():
        var_list=dl.sel_vars()    
        for s in sample_list:
            #print(dfs[s].columns)
            df_trans = dfs[s][var_list]
            #print("before:\n",df_trans.head())
            df_trans = sc.transform(df_trans)
            #print("after transformation:\n",df_trans[:5])
            predictScore = model.predict(df_trans)
            dfs[s].loc[:,'score'] = dfs[s].loc[:,'Njets']
            dfs[s].loc[:,'score'] = predictScore
            #print("with score:",dfs[s].head())

        for i in range(3,10):
            print(i/10)
            pl.plot_var(dfs,sample_list,'mjj',True,i/10)
            pl.plot_var(dfs,sample_list,'Njets',True,i/10)
            pl.plot_var(dfs,sample_list,'score',True,i/10)
            
    elif process_type == 'read' or process_type == 'train':

        if dfs:
            
            print("prepare for training:")
            print(" - transform input :")
            cat_list=['l0_id','l1_id','dileptype','mjjctag']
            noncat_list = list(set(dfs['ttW'].columns)-set(cat_list))
            for s in sample_list:
                dfs[s]=hp.val_to_cat(dfs[s],cat_list)
                dfs[s][noncat_list]=hp.norm_gev(dfs[s][noncat_list])

            var_list= list(dfs['ttW'].columns)
            print(var_list)
            print(" - split samples to train/test features/targets :")  
            X_train, X_test, y_train, y_test, w_train, w_test  = hp.pred_ds(dfs)

            learning_rate = 0.001
            nepochs = 500
            batch_size = 512
            validation_split = 0.2

            if process_type == 'train':
                
                model = md.create_model(learning_rate,var_list)
                epochs, hist = md.train_model(model, X_train, y_train, w_train,
                                           nepochs, batch_size, validation_split)

            
                list_of_metrics_to_plot = ['loss','val_loss']
                hp.plot_curve(epochs, hist, list_of_metrics_to_plot)
                list_of_metrics_to_plot = ['acc','val_acc']
                hp.plot_curve(epochs, hist, list_of_metrics_to_plot)

                print("\n Train set:")
                score_tr = model.evaluate(X_train, y_train, batch_size=batch_size)  
                print(score_tr)
                print("\n Evaluate the new model against the test set:")
                score = model.evaluate(X_test, y_test, batch_size=batch_size)  
                print(score)

                model.save('Outputs/training/model_nn_v0.h5')
            elif process_type == 'read':
                model = load_model('Outputs/training/model_nn_v0.h5')

            if model and (process_type != 'apply'):
                testPredict = model.predict(X_test)
                xt_p = {}
                x_sig,x_bkg =hp.sig_bkg_ds_separate(X_test,y_test)
                xt_p['ttW'] = model.predict(x_sig)
                xt_p['ttbar'] = model.predict(x_bkg)
                x_p = {}
                x_sig,x_bkg =hp.sig_bkg_ds_separate(X_train,y_train)
                x_p['ttW'] = model.predict(x_sig)
                x_p['ttbar'] = model.predict(x_bkg)
                
                bins = [i/80 for i in range(80)]
                bins.append(1.)

                plt.figure("response")
                for i in sample_list:                    
                    plt.hist(xt_p[i], bins, alpha=0.5, label=i+' Predict', density=True, color=samples[i]['color'])
                    plt.hist(x_p[i], bins, alpha=1, label=i+' Train', density=True, color=samples[i]['color'], histtype='step')
                plt.legend(loc="upper right")
                plt.savefig("Outputs/training/classPred_NN_ttw_ttbar.png", transparent=True)
                plt.close("response")
                
                print( classification_report(y_test, testPredict.round(), target_names=["ttbar", "ttW"]))# 
                auc = roc_auc_score(y_test, testPredict)
                print( "Area under ROC curve: %.4f"%(auc))
                hp.get_roc(y_test,testPredict)

                print_summary = True
                if print_summary:
                    with open("Outputs/training/summary.txt", "w") as f:
                        f.write("Parameters:\n")
                        #f.write("         classifier_model: {}\n".format(model.get_config()))
                        f.write("LR {}, epochs {}, batch_size {}, VS {} \n".format(learning_rate,nepochs,batch_size,validation_split))
                        f.write(": {}\n".format(classification_report(y_test, testPredict.round(), target_names=["signal", "background"])))                        
                        f.write("\nAUC:{}\n".format(auc))
                        f.write("model.summary() :{}\n".format(model.summary()) )
                    

if __name__ == "__main__":
    main() 
