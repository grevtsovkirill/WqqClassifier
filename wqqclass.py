import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

from samples import *
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from keras.layers import Layer, Input, Dense, Dropout
    from keras.models import Sequential, load_model
    from keras.callbacks import EarlyStopping, ModelCheckpoint


def sel_vars(list_name="varlist.json"):
    with open(list_name) as vardict:
        variablelist = json.load(vardict)[:]

    print(variablelist)
    return variablelist

def data_load(in_list, do_clean=doclean):
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
            if do_clean:
                df[s] = df[s].loc[df[s].region==0]
                df[s] = df[s][var_list]
        else:
            print(s,' is not in available sample list')
            break
    return df

    

def pred_ds(dfs,test_samp_size=0.33):    
    X = np.concatenate((dfs['ttW'],dfs['ttbar']))
    sc = StandardScaler(copy=False)
    X = sc.fit_transform(X)
    with open('Outputs/training/scaler.pickle', 'wb') as f:
        pickle.dump(sc, f)
    y = np.concatenate((np.ones(dfs['ttW'].shape[0]),np.zeros(dfs['ttbar'].shape[0]))) # class lables
    class_weight = len(dfs['ttW'])/len(dfs['ttbar'])
    print("class_weight ", class_weight)
    w = np.concatenate(( [1]*(dfs['ttW'].shape[0]),[class_weight]*(dfs['ttbar'].shape[0]))) # class lables                                                                       
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size = test_samp_size)
    return X_train, X_test, y_train, y_test, w_train, w_test


def sig_bkg_ds_separate(X, y,key="Predict"):
    el,cnt = np.unique(y,return_counts=True)
    print(el,cnt)
    
    xt_sig = np.zeros(shape=(cnt[1],len(X[0])))
    xt_bkg = np.zeros(shape=(cnt[0],len(X[0])))
    i_s=0
    i_b=0
    for i in range(len(y)):
        if y[i]==samples['ttW']['class']:
            xt_sig[i_s]=X[i]
            i_s+=1
        else:
            xt_bkg[i_b]=X[i]
            i_b+=1
            
    print(i_s,i_b)
    return xt_sig, xt_bkg




def main():
    print("load data")
    dfs=data_load(sample_list)
    print(dfs['ttW'].columns)
    if process_type =='plot':
        pl.plot_var(dfs,sample_list,'Njets')
        pl.plot_var(dfs,['ttW','ttbar'],'Njets',False)

    elif process_type == 'apply':
        print("apply mode")
        model = load_model('Outputs/training/model_nn_v0.h5')
        with open('Outputs/training/scaler.pickle', 'rb') as f:
            sc = pickle.load(f)

            #def model_create_feature():
        var_list=sel_vars()    
        for s in sample_list:
            #print(dfs[s].columns)
            df_trans = dfs[s][var_list]
            #print("before:\n",df_trans.head())
            df_trans = sc.transform(df_trans)
            #print("after transformation:\n",df_trans[:5])
            predictScore = model.predict(df_trans)
            dfs[s].loc[:,'score'] = dfs[s].loc[:,'Njets']
            dfs[s].loc[:,'score'] = predictScore
            #print(dfs[s].head())

        for i in range(3,10):
            print(i/10)
            pl.plot_var(dfs,sample_list,'mjj',True,i/10)
            
    elif process_type == 'read' or process_type == 'train':

        if dfs:
            print("prepare for training, ")
            X_train, X_test, y_train, y_test, w_train, w_test  = pred_ds(dfs)

            learning_rate = 0.01
            nepochs = 500
            batch_size = 256
            validation_split = 0.2

            if process_type == 'train':
                var_list=sel_vars() 
                model = md.create_model(learning_rate,var_list)        
                epochs, hist = md.train_model(model, X_train, y_train, w_train,
                                           nepochs, batch_size, validation_split)

            
                print("\n Evaluate the new model against the test set:")
                print(model.evaluate(X_test, y_test, batch_size=batch_size))
                list_of_metrics_to_plot = ['loss','val_loss']
                hp.plot_curve(epochs, hist, list_of_metrics_to_plot)
                list_of_metrics_to_plot = ['acc','val_acc']
                hp.plot_curve(epochs, hist, list_of_metrics_to_plot)

                model.save('Outputs/training/model_nn_v0.h5')
            elif process_type == 'read':
                model = load_model('Outputs/training/model_nn_v0.h5')

            if model and (process_type != 'apply'):
                testPredict = model.predict(X_test)
                xt_p = {}
                x_sig,x_bkg =sig_bkg_ds_separate(X_test,y_test)
                xt_p['ttW'] = model.predict(x_sig)
                xt_p['ttbar'] = model.predict(x_bkg)
                x_p = {}
                x_sig,x_bkg =sig_bkg_ds_separate(X_train,y_train)
                x_p['ttW'] = model.predict(x_sig)
                x_p['ttbar'] = model.predict(x_bkg)
                
                bins = [i/40 for i in range(40)]
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
