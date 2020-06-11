import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

import pandas as pd
import numpy as np
from samples import *

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
