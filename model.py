from keras.layers import Layer, Input, Dense, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

import pandas as pd

def create_model(my_learning_rate,var_list):
    dense_dim=len(var_list)
    model = Sequential()
    model.add(Dense(dense_dim, input_dim=dense_dim, activation='relu'))
    #model.add(Dense(256, activation='relu'))    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.3, noise_shape=None, seed=None))
    #model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))    
    model.add(Dense(32, activation='relu'))    
    #model.add(Dense(20, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid',name='classifier_output'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, train_features, train_label, weights,
                epochs, batch_size=None, validation_split=0.1):

    
    earlyStop = EarlyStopping(monitor='val_loss', verbose=True, patience=30)

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
