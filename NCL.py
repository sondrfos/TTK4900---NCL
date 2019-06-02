#!/usr/bin/env python
# coding: utf-8


from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, concatenate
from keras.layers.merge import concatenate
from keras.utils import plot_model, to_categorical
from keras.datasets import mnist, cifar10
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


import keras.backend as K
import keras.losses
import numpy as np
import pandas as pd
import datetime
import sys
import logging
import os
import math
import time
import tensorflow as tf



def load_equinor(test_size, dataset_size):
    dataset_df = pd.read_hdf('./full_dataset.h5')
    print(dataset_df.shape)
    dataset_df = dataset_df.sample(frac = dataset_size, random_state=1)
    print(dataset_df.shape)
    
    #hente ut fasit
    y = np.array(dataset_df['is_iceberg'])
    dataset_df = dataset_df.drop(columns = ['is_iceberg'])
    
    #split i train/test, med samme seed hver gang
    X_train_df, X_test_df, y_train, y_test = train_test_split(dataset_df, y, test_size = test_size, random_state = 42) 
    
    #hente ut ID
    id_array = np.array(X_test_df['id'])
    
    #hente ut bands all band 1 i en variabel og alle band 2 i en annen
    band_1 = np.array([band.astype(np.float32).reshape(HEIGHT, WIDTH) for band in X_train_df['band 1']])
    band_2 = np.array([band.astype(np.float32).reshape(HEIGHT, WIDTH) for band in X_train_df['band 2']])
    band_3 = (band_1/band_2)
    
    #slå sammen til en array før split
    X_train = np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis], band_3[:, :, :, np.newaxis]], axis=-1)
    
    #hente ut bands all band 1 i en variabel og alle band 2 i en annen
    band_1_test = np.array([band.astype(np.float32).reshape(HEIGHT, WIDTH) for band in X_test_df['band 1']])
    band_2_test = np.array([band.astype(np.float32).reshape(HEIGHT, WIDTH) for band in X_test_df['band 2']])
    band_3_test = (band_1_test/band_2_test)
    
    X_test = np.concatenate([band_1_test[:, :, :, np.newaxis], band_2_test[:, :, :, np.newaxis], band_3_test[:, :, :, np.newaxis]], axis=-1)
    
    #returner alt
    return id_array, X_train, y_train, X_test, y_test
    
def load_cifar10(test_size, dataset_size, class_num):
    
    (X_train_all, y_train_all), (X_test_all, y_test_all) = cifar10.load_data()
    dataset_size = int(dataset_size*6000)
    X_mod, y_mod = [], []
    for i, y in enumerate(y_train_all):
        X_mod.append(X_train_all[i])
        if y == class_num:
            y_mod.append(1)
        else:
            y_mod.append(0)
    for i, y in enumerate(y_test_all):
        X_mod.append(X_test_all[i])
        if y == class_num:
            y_mod.append(1)
        else:
            y_mod.append(0)
    dataset_df = pd.DataFrame({'X': X_mod, 'y':y_mod})
    # Divide by class
    class_0_df = dataset_df[dataset_df['y'] == 0]
    class_1_df = dataset_df[dataset_df['y'] == 1]
    
    class_0_df_sampled = class_0_df.sample(dataset_size)
    class_1_df_sampled = class_1_df.sample(dataset_size)
    dataset_df = pd.concat([class_0_df_sampled, class_1_df_sampled], axis=0)
    
    #hente ut fasit
    y = np.array(dataset_df['y'])
    dataset_df = dataset_df.drop(columns = ['y'])
    
    #split i train/test, med samme seed hver gang
    X_train_df, X_test_df, y_train, y_test = train_test_split(dataset_df, y, test_size = test_size, random_state = 42) 
    X_train = np.array([X.astype(np.float32).reshape(HEIGHT, WIDTH, CHANNELS) for X in X_train_df['X']])
    X_test = np.array([X.astype(np.float32).reshape(HEIGHT, WIDTH, CHANNELS) for X in X_test_df['X']])
    #returner alt
    return X_train, y_train, X_test, y_test


def custom_loss(regularization):
    def loss(y_true, y_pred):
        average_pred = K.mean(y_pred, axis = 1)
        correlation = 0
        for i in range(K.int_shape(y_pred)[1]):
            if i == MODEL_NUM-1:
                own_error = (y_pred[:,i] - average_pred)
            else:
                correlation += (y_pred[:,i] - average_pred)
        correlation *= own_error
        return K.mean(K.square(1/2*(y_pred[:,MODEL_NUM-1:MODEL_NUM] - y_true)), axis=-1) + K.mean(regularization*correlation, axis=-1)
    return loss

def read_model_preds(model_names, regularization):
    num_models = len(model_names)
    load_path = MODELS_LOAD_PATH + 'val_preds/'
    for model_name in model_names:
        load_path += model_name
    load_path += ', regularization:' + str(regularization) + '.h5'
    df_results = pd.read_hdf(load_path)
    #print(df_results.shape)
    preds = np.empty((num_models, 3466))
    for i in range(num_models):
        #print(df_results['pred'+str(i)])
        preds[i] = np.array([pred for pred in df_results['pred'+str(i)]])
    ys = np.array([y for y in df_results['y']])
    return preds, ys

def save_log():
    sys.stdout = open(MODEL_PATH + 'log.txt', 'a+')
    my_stderr = sys.stderr = open(MODEL_PATH + 'log.txt', 'a+')  # redirect stderr to file
    get_ipython().log.handlers[0].stream = my_stderr  # log errors to new stderr
    get_ipython().log.setLevel(logging.INFO)  # errors are logged at info level
    
def callbacks(model_name):
    return [
        EarlyStopping(monitor='val_loss', patience=7, verbose=1, min_delta=1e-5, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_delta=1e-5, mode='min'),
        ModelCheckpoint(monitor='val_loss', filepath=MODEL_PATH + 'weights/' + model_name + '.hdf5',
                        save_best_only=True, save_weights_only=True, mode='min')
    ]


def build_cnn(conv_layers, dense_layers, conv_dropout):
    raw_input = Input(shape=(HEIGHT,WIDTH,CHANNELS))
    norm_input = BatchNormalization()(raw_input)
    layer = Conv2D(conv_layers[0], kernel_size=(3, 3), padding = 'same', activation='relu', name='conv0')(norm_input)
    #layer = Conv2D(conv_layers[0], kernel_size=(3, 3), activation='relu', name='conv0')(norm_input)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)
    if conv_dropout:
        layer = Dropout(conv_dropout)(layer)
    i=1
    for conv_layer_size in conv_layers[1:]:
        layer = Conv2D(conv_layer_size, kernel_size=(3, 3), padding = 'same', activation='relu', name='conv'+str(i))(layer)
        #layer = Conv2D(conv_layer_size, kernel_size=(3, 3), activation='relu', name='conv'+str(i))(layer)
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)
        i+=1
        if conv_dropout:
            layer = Dropout(conv_dropout)(layer)

    layer = Flatten()(layer)

    for dense_layer_size in dense_layers:
        layer = Dense(dense_layer_size, activation='relu')(layer)
        if conv_dropout:
            layer = Dropout(conv_dropout)(layer)

    output = Dense(1, activation='sigmoid')(layer)
    optimizer = RMSprop(lr=0.0002)
    model = Model(inputs=raw_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def build_stacker(num_models, top_layers, dropout):
    raw_input = Input(shape=(num_models, NUM_CLASSES))
    norm_input = BatchNormalization()(raw_input)
    layer = Flatten()(norm_input)
    for top_layer in top_layers:
        layer = Dense(top_layer, activation='relu')(layer)
        if dropout:
            layer = Dropout(dropout)(layer)
            
    output = Dense(1, activation='sigmoid')(layer)
    model = Model(inputs=raw_input, outputs=output)
    optimizer = RMSprop(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def combine_models(models):
    raw_input = Input(shape=(HEIGHT,WIDTH,CHANNELS))
    model_output = ['']*len(models)
    for i, model in enumerate(models):
        model_output[i] = model(raw_input)
    output = concatenate(model_output)
    model = Model(inputs = raw_input, output = output)
    return model

def train_submodel(model, submodel_name, loss, optimizer, X_train, y_train, X_val, y_val):
    #make only the submodel trainable
    changed = False
    for layer in model.layers:
        if layer.name == submodel_name:
            if layer.trainable == True:
                break
            else:
                layer.trainable = True
                changed = True
        else:
            layer.trainable = False
    print('Training ' + submodel_name)
    if changed: model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    #train the submodel, and return the loss
    history = model.fit(X_train, y_train, epochs = 1, batch_size = 32, validation_data = (X_val, y_val), verbose = 2)
    return history.history['val_loss'], model
    



MODEL_NUM = 1
HEIGHT = 32
WIDTH = 32
CHANNELS = 3
NUM_CLASSES = 1
NUM_FOLDS = 4
PRED_CLIP = 0.00001 
TEST_SIZE = 0.3
DATASET_SIZE = 0.1
TIME = datetime.datetime.now()
MODEL_PATH = './ncl/' + str(TIME) + '/'
MODELS_LOAD_TIME = '2019-03-24 22:02:27.926678'
MODELS_LOAD_PATH = './ncl/' + MODELS_LOAD_TIME + '/'
if not os.path.exists(os.path.dirname(MODEL_PATH)):
    os.makedirs(os.path.dirname(MODEL_PATH))
    os.makedirs(os.path.dirname(MODEL_PATH+'weights/'))
    os.makedirs(os.path.dirname(MODEL_PATH+'val_preds/'))
save_log()
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

conv_layers_to_try = [
    [64, 128, 256],
    [32, 64, 128, 256]
]
dense_layers_to_try = [
    #[128, 64],
    #[256, 128],
    [512, 256]
]
dropout_to_try = [0.2]#[0, 0.2]
configs = []
for conv_layers in conv_layers_to_try:
    for dense_layers in dense_layers_to_try:
        for dropout in dropout_to_try:
            config = dict(conv_layers = conv_layers,
                          dense_layers = dense_layers,
                          dropout = dropout)
            configs.append(config)
            
regularizations = [5e-07]
num_models = len(configs)
num_runs = [0,1,2,3,4,5,6,7,8,9]
class_num = 5
epochs = 100
patience = 7

model_names = ['']*num_models
for i, config in enumerate(configs):
    #name models
    model_names[i] = 'conv_layers: ' + str(config['conv_layers'])+ ', dense_layers: ' + str(config['dense_layers']) +', dropout: ' + str(config['dropout'])

#import data
#id_array, X_train, y_train, X_test, y_test = load_equinor(TEST_SIZE, DATASET_SIZE)
X_train, y_train, X_test, y_test = load_cifar10(TEST_SIZE, DATASET_SIZE, class_num)
folds = list(StratifiedKFold(n_splits = NUM_FOLDS, shuffle = True, random_state = 1).split(X_train, y_train))

#pint a summary for the top of the log-file
print('TRAINING ON CIFAR10 CLASS_NUM = ' + str(class_num) + " " + str(NUM_FOLDS) + '-FOLD CV FOR REGULARIZATION = ' + str(regularizations) + ' ' + str(num_runs) + ' TIMES USING MODEL ARCHITECTURES')
#print('TRAINING ON C-CORE/EQUINOR = ' + str(NUM_FOLDS) + '-FOLD CV FOR REGULARIZATION = ' + str(regularizations) + ' ' + str(num_runs) + ' TIMES USING MODEL ARCHITECTURES')

for name in model_names:
    print(name)
print('\n')

######### NCL ##########
for regularization in regularizations:
    for run in num_runs:
        total_val_pred = [None]*num_models
        total_val_y = [None]*num_models
        start = time.time()
        print(start)

        #k-fold cross validation
        for j, (train_idx, val_idx) in enumerate(folds):
            optimizer = RMSprop(lr=0.0002)
            loss = custom_loss(regularization)
            
            #format data
            X_train_cv = X_train[train_idx]
            y_train_cv = y_train[train_idx]
            X_val_cv = X_train[val_idx]
            y_val_cv = y_train[val_idx]

            lowest_loss = [np.Inf]*num_models
            patience_counter = [0]*num_models
            models = ['']*num_models
            for i, config in enumerate(configs):
                models[i] = build_cnn(config['conv_layers'], config['dense_layers'], config['dropout'])
                models[i].name = 'model'+str(i)
                
            #combine the models so they can be trained depending on each other
            model = combine_models(models)
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            plot_model(model, to_file=MODEL_PATH+'model_plot.png', show_shapes=True, show_layer_names=True)
            model_save_name = 'run: ' + str(run)  + ', regularization: ' + str(regularization)
            model_save_name_fold = model_save_name + ', fold: ' + str(j)

            #loop through training procedure
            for epoch in range(epochs):
                print('################ EPOCH ' + str(epoch) + ' ################')

                #train each model and check improvements
                for model_idx in range(num_models):
                    
                    #only train if we haven't lost patience on this model
                    if patience_counter[model_idx] < patience:
                        MODEL_NUM = model_idx + 1                       
                        val_loss, model = train_submodel(model=model, submodel_name='model' + str(model_idx), loss=loss, optimizer=optimizer,
                                                   X_train=X_train_cv, y_train=y_train_cv, X_val=X_val_cv, y_val=y_val_cv)
                        
                        #check if this resulted in an improvement, then save the model. If not, roll back
                        if val_loss[0] > lowest_loss[model_idx]:
                            patience_counter[model_idx] += 1
                            print('My patience is now ' + str(patience_counter))
                            while patience_counter[model_idx] < patience:
                                val_loss, model = train_submodel(model=model, submodel_name='model' + str(model_idx), loss=loss, optimizer=optimizer,
                                                   X_train=X_train_cv, y_train=y_train_cv, X_val=X_val_cv, y_val=y_val_cv)
                                if val_loss[0] < lowest_loss[model_idx]: break
                                else: 
                                    patience_counter[model_idx] += 1
                                    print('My patience is now ' + str(patience_counter))
                        
                        if val_loss[0] < lowest_loss[model_idx]:
                            print('Lower loss model' + str(model_idx) + '. It was ' + str(lowest_loss[model_idx]) + ' and is now ' + str(val_loss[0]))
                            lowest_loss[model_idx] = val_loss[0]
                            patience_counter[model_idx] = 0
                            model.save(MODEL_PATH + 'weights/' + model_save_name_fold + '.h5')
                        else:
                            print('Loading old model, since no improvement was found')
                            del model
                            model = load_model(filepath=MODEL_PATH + 'weights/' + model_save_name_fold + '.h5', custom_objects={'loss': loss})
                        del val_loss
                        
                #break training if we have lost our patience
                if patience_counter >= [patience]*num_models:
                    print('lost patience')
                    break;
            for i in range(num_models):
                del models[0]
            for i in range(num_models):
                for layer in model.layers:
                    if layer.name == 'model'+str(i):
                        print('Extract model '+str(i))
                        models.append(layer)
                models[i].compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                plot_model(models[i], to_file=MODEL_PATH+'model'+str(i)+'_plot.png', show_shapes=True, show_layer_names=True)
                models[i].save(MODEL_PATH + 'weights/run: ' + str(run) + ', ' + model_names[i] + 'regularization: ' + str(regularization) +  ', fold:' + str(j) +  '.h5')

                #predict for test and val fold 
                train_pred = models[i].predict(X_train_cv).astype(np.float64)
                val_pred = models[i].predict(X_val_cv).astype(np.float64)

                if total_val_pred[i] is not None:
                    total_val_pred[i] = np.append(total_val_pred[i], val_pred)
                    total_val_y[i] = np.append(total_val_y[i], y_val_cv)
                else:
                    total_val_pred[i] = val_pred
                    total_val_y[i] = y_val_cv
                print(total_val_pred[i].shape)

                print('Train score for model'+str(i)+': ', log_loss(y_train_cv, train_pred, eps = PRED_CLIP), '\n')
                print('Val score for model'+str(i)+': ', log_loss(y_val_cv, val_pred, eps = PRED_CLIP), '\n')
            for i in range(num_models):
                del models[0]
            del model
            K.clear_session()

        #save validation_set prediction to file
        dataframes = [0]*len(total_val_pred[0])
        temp = [0]*(num_models+1)
        for sample_idx in range(len(total_val_pred[0])):
            for model_idx in range(num_models):
                temp[model_idx] = pd.DataFrame({'pred' + str(model_idx): [total_val_pred[model_idx][sample_idx]]}, index = [sample_idx])
            temp[num_models] = pd.DataFrame({'y': [total_val_y[0][sample_idx]]}, index = [sample_idx])
            dataframes[sample_idx] = pd.concat(temp, axis=1)
        df_val = pd.concat(dataframes)
        print(df_val.head())
        print(df_val.shape)
        df_val.to_hdf(MODEL_PATH + 'val_preds/' + model_save_name + '.h5', key = 'df', mode = 'w')
        end = time.time()
        print(end)
        print(end - start)
