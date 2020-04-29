import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from samples import *


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
    from sklearn.metrics import roc_auc_score, roc_curve, auc
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from keras.layers import Layer, Input, Dense, Dropout
    from keras.models import Sequential, load_model
    from keras.callbacks import EarlyStopping, ModelCheckpoint


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
                df[s] = df[s].loc[df[s].region==0]
                df[s] = df[s][var_list]
        else:
            print(s,' is not in available sample list')
            break
    return df

def plot_var(df_bkg,lab_list,var,do_stack=True,GeV=1):
    stack_var=[]
    stack_var_w=[]
    stack_var_leg=[]
    stack_var_yields=[]
    stack_var_col=[]
    for i in lab_list:
        stack_var.append(df_bkg[i][var].loc[df_bkg[i].region==0]*GeV)
        stack_var_w.append(df_bkg[i].weight_tot.loc[df_bkg[i].region==0])
        stack_var_yields.append(df_bkg[i].weight_tot.loc[df_bkg[i].region==0].sum())
        yield_val = '{0:.2f}'.format(df_bkg[i].weight_tot.loc[df_bkg[i].region==0].sum())
        print(i," ", yield_val)
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
        plt.ylim(1e-1, 1e8)
        plt.ylabel('# Events',fontsize=12) 
        plt.legend()
        plt.savefig('Outputs/stack/'+var+'.png', transparent=True)
        plt.close("stack")
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
    

def pred_ds(dfs,test_samp_size=0.33):    
    X = np.concatenate((dfs['ttW'],dfs['ttbar']))
    sc = StandardScaler(copy=False)
    X = sc.fit_transform(X)
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

def create_model(my_learning_rate):
    dense_dim=len(sel_vars())
    model = Sequential()
    model.add(Dense(dense_dim, input_dim=dense_dim, activation='relu'))
    model.add(Dense(30, activation='relu'))    
    #model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    #model.add(Dropout(rate=0.005, noise_shape=None, seed=None))
    model.add(Dense(1, activation='sigmoid',name='classifier_output'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, train_features, train_label, weights,
                epochs, batch_size=None, validation_split=0.1):

    
    earlyStop = EarlyStopping(monitor='val_loss', verbose=True, patience=10)

    nn_mChkPt = ModelCheckpoint('Outputs/training/nn_weights.h5',monitor='val_loss', verbose=True,
                                  save_best_only=True,
                                  save_weights_only=True)
    
    history = model.fit(x=train_features, y=train_label,
                        sample_weight=weights,
                        batch_size=batch_size,
                        epochs=epochs, shuffle=True, 
                        validation_split=validation_split,
                        callbacks=[earlyStop, nn_mChkPt]
    )
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

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


def main():
    print("load data")
    dfs=data_load(sample_list)
    #print(dfs['ttW'].columns)
    if process_type =='plot':
        plot_var(dfs,sample_list,'Njets')
        plot_var(dfs,['ttW','ttbar'],'Njets',False)

    elif process_type == 'apply':
        print("apply mode")
        model = load_model('Outputs/training/model_nn_v0.h5')


    elif process_type == 'read' or process_type == 'train':

        if dfs:
            print("prepare for training, ")
            X_train, X_test, y_train, y_test, w_train, w_test  = pred_ds(dfs)

            learning_rate = 0.001
            nepochs = 500
            batch_size = 32
            validation_split = 0.2

            if process_type == 'train':

                model = create_model(learning_rate)        
                epochs, hist = train_model(model, X_train, y_train, w_train,
                                           nepochs, batch_size, validation_split)

            
                print("\n Evaluate the new model against the test set:")
                print(model.evaluate(X_test, y_test, batch_size=batch_size))
                list_of_metrics_to_plot = ['loss','val_loss']
                plot_curve(epochs, hist, list_of_metrics_to_plot)
                list_of_metrics_to_plot = ['acc','val_acc']
                plot_curve(epochs, hist, list_of_metrics_to_plot)

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
                
                print( classification_report(y_test, testPredict.round(), target_names=["signal", "background"]))
                auc = roc_auc_score(y_test, testPredict)
                print( "Area under ROC curve: %.4f"%(auc))
                get_roc(y_test,testPredict)

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
