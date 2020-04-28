import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from samples import *


seed=8
np.random.seed(seed)



import argparse
parser = argparse.ArgumentParser(description='Prepare classifier')
parser.add_argument('-t','--type', required=True, type=str, choices=['plot', 'train','apply'], help='Choose processing type: explore variable [plot], train the model [train], or apply existing model [apply] ')
parser.add_argument('-s','--samples', nargs='+', default=['ttW','ttbar'], help='Choose list of samples to run over ')
parser.add_argument('-c','--clean', default=False, help='Use selected list of variables ')

args = parser.parse_args()

process_type = vars(args)["type"]
sample_list = vars(args)["samples"]
doclean = vars(args)["clean"]

if process_type =='train' or process_type == 'apply' :
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from keras.layers import Layer, Input, Dense, Dropout
    from keras.models import Sequential, load_model


scale_to_GeV=0.001
binning = {"DRll01": np.linspace(-2, 6, 24),
           "max_eta": np.linspace(0, 2.5, 26),
           "Njets": np.linspace(0, 10, 10),
          }

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
    

def pred_ds(dfs,test_samp_size=0.33):    
    X = np.concatenate((dfs['ttW'],dfs['ttbar']))
    sc = StandardScaler()
    X = sc.fit_transform(X)
    y = np.concatenate((np.ones(dfs['ttW'].shape[0]),np.zeros(dfs['ttbar'].shape[0]))) # class lables                                                                       
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_samp_size)
    return X_train, X_test, y_train, y_test

def create_model(my_learning_rate):
    dense_dim=len(sel_vars())
    model = Sequential()
    model.add(Dense(dense_dim, input_dim=dense_dim, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(35, activation='relu'))
    model.add(Dropout(rate=0.1, noise_shape=None, seed=None))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid',name='classifier_output'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True, 
                      validation_split=validation_split)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

def plot_curve(epochs, hist, list_of_metrics,save=True):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    
    plt.ylabel('cross-entropy loss',fontsize=14)
    plt.xlabel('epochs',fontsize=14)
    plt.legend()
    if save:
        plt.savefig("Plots/training/loss_NNw.png", transparent=True)
    else:
        plt.show()

def get_roc(y_test, y_predicted):
    fpr, tpr, _ = roc_curve(y_test, y_predicted)
    plt.figure()
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
    plt.savefig("Plots/training/ROC_NN_ttw_ttbar.png", transparent=True)
    


def main():
    print("load data")
    dfs=data_load(sample_list)
    #print(dfs['ttW'].columns)
    if process_type =='plot':
        plot_var(dfs,['ttW','ttZmumu','ttbar'],'Njets')
        plot_var(dfs,['ttW','ttbar'],'Njets',False)

    else:
        if dfs:
            print("prepare for training, ")
            X_train, X_test, y_train, y_test = pred_ds(dfs)

            if process_type == 'train':
                learning_rate = 0.003
                epochs = 500
                batch_size = 16000
                validation_split = 0.2

                model = create_model(learning_rate)        
                epochs, hist = train_model(model, X_train, y_train, 
                                           epochs, batch_size, validation_split)

            
                print("\n Evaluate the new model against the test set:")
                print(model.evaluate(X_test, y_test, batch_size=batch_size))
                
                list_of_metrics_to_plot = ['loss','val_loss']
                plot_curve(epochs, hist, list_of_metrics_to_plot)
                model.save('Models/nn_v0.h5')
            elif process_type == 'apply':
                model = load_model('Models/nn_v0.h5')

            if model:
                testPredict = model.predict(X_test)
                get_roc(y_test,testPredict)
                print( classification_report(y_test, testPredict.round(), target_names=["signal", "background"]))
                print( "Area under ROC curve: %.4f"%(roc_auc_score(y_test, testPredict)))


if __name__ == "__main__":
    main() 
