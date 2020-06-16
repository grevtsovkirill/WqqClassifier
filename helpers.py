import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

import pandas as pd
import numpy as np
from samples import *

def sig_bkg_ds_separate(X, y,key="Predict"):
    print("in sig_bkg_ds_separate: ")
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

def val_to_cat(df, fnames):
    if len(fnames)==1:
        df = pd.concat([df,pd.get_dummies(df[fnames], prefix=fnames)],axis=1)
    else:
        for i in fnames:
            df = pd.concat([df,pd.get_dummies(df[i], prefix=i)],axis=1)
    df = df.drop(fnames, axis=1)
    return df

def norm_gev(df):
    df_max = df.max()
    df_mean = df.mean()
    df_std = df.std()
    #df_norm = (df/df_max)
    df_norm = (df-df_mean)/df_std
    return df_norm

def pred_ds(dfs,cat_list,noncat_list,test_samp_size=0.33):    
    dfs['ttW']['target']=np.ones(dfs['ttW'].shape[0])
    dfs['ttbar']['target']=np.zeros(dfs['ttbar'].shape[0])

    data = pd.concat([dfs['ttW'],dfs['ttbar']], ignore_index=True, sort=False)
    data = val_to_cat(data,cat_list) 
    y_tot = data.pop('target')
    #print("with target",data.head())
    #print("with target",data.tail())
    
    #X = np.concatenate((dfs['ttW'],dfs['ttbar']))
    #sc = StandardScaler(copy=False)
    #X = sc.fit_transform(X)
    #with open('Outputs/training/scaler.pickle', 'wb') as f:
     #   pickle.dump(sc, f)
    #y = np.concatenate((np.ones(dfs['ttW'].shape[0]),np.zeros(dfs['ttbar'].shape[0]))) # class lables
    class_weight = len(dfs['ttW'])/len(dfs['ttbar'])
    print("class_weight ", class_weight)
    #w = np.concatenate(( [1]*(dfs['ttW'].shape[0]),[class_weight]*(dfs['ttbar'].shape[0]))) # class lables                                                                       
    #X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size = test_samp_size)
    X_train, X_test, y_train, y_test = train_test_split(data, y_tot, test_size = test_samp_size)
    #sc = StandardScaler(copy=False)
    ct = ColumnTransformer([('sc', StandardScaler(), noncat_list)], remainder='passthrough')
    #ct = ColumnTransformer([('sc', MinMaxScaler(), noncat_list)], remainder='passthrough')
    print(X_train.head())
    varlist = list(X_train.columns)  
    with open('Outputs/training/vl.pickle', 'wb') as f:
        pickle.dump(varlist, f)
    X_train = ct.fit_transform(X_train)
    with open('Outputs/training/scaler.pickle', 'wb') as f:
        pickle.dump(ct, f)
    print("after transform",X_train[0])
    X_test = ct.transform(X_test)

    #return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), class_weight  #, w_train, w_test
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), class_weight, varlist  #, w_train, w_test



def plot_curve(epochs, hist, list_of_metrics,save=True):
    plt.figure("loss")
    label_val = list_of_metrics[0]
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)    
    plt.ylabel(label_val,fontsize=14)
    plt.xlabel('epochs',fontsize=14)
    plt.legend()
    if save:
        plt.savefig("Outputs/training/"+label_val+"_NNw.png", transparent=True)
    else:
        plt.show()
    plt.close("loss")

def get_roc(y_test, y_predicted):
    fpr, tpr, _ = roc_curve(y_test, y_predicted)
    plt.figure("roc")
    lw = 2
    plt.plot(tpr, 1-fpr, 
             lw=lw, label='%s ROC (%0.3f)' % ("NN ", roc_auc_score(y_test, y_predicted))) #color='darkorange',
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    #plt.plot(fpr, tpr, 
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.xlabel("Signal efficiency")
    plt.ylabel("Background rejection")
    plt.title('Background rejection versus Signal efficiency')
    plt.legend(loc="lower left")
    plt.savefig("Outputs/training/ROC_NN_ttw_ttbar.png", transparent=True)
    plt.close("roc")
