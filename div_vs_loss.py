#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import pycuda.autoinit
import pycuda.driver as cuda
import tensorflow as tf
import sys
import cv2
import time

from matplotlib import pyplot as plt

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from scipy.ndimage.interpolation import zoom

from IPython.display import clear_output

from keras import backend as K
from keras.backend import tensorflow_backend
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers.core import Lambda
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.python.framework import ops



def load_equinor(test_size):
    dataset_df = pd.read_hdf('./full_dataset.h5')
    
    #hente ut fasit
    y = np.array(dataset_df["is_iceberg"])
    dataset_df = dataset_df.drop(columns = ["is_iceberg"])
    
    #split i train/test, med samme seed hver gang
    X_train_df, X_test_df, y_train, y_test = train_test_split(dataset_df, y, test_size = test_size, random_state = 42) 
    
    #hente ut ID
    id_array = np.array(X_test_df["id"])
    
    #hente ut bands all band 1 i en variabel og alle band 2 i en annen
    band_1 = np.array([band.astype(np.float32).reshape(HEIGHT, WIDTH) for band in X_train_df["band 1"]])
    band_2 = np.array([band.astype(np.float32).reshape(HEIGHT, WIDTH) for band in X_train_df["band 2"]])
    band_3 = (band_1/band_2)
    
    #slå sammen til en array før split
    X_train = np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis], band_3[:, :, :, np.newaxis]], axis=-1)
    
    #hente ut bands all band 1 i en variabel og alle band 2 i en annen
    band_1_test = np.array([band.astype(np.float32).reshape(HEIGHT, WIDTH) for band in X_test_df["band 1"]])
    band_2_test = np.array([band.astype(np.float32).reshape(HEIGHT, WIDTH) for band in X_test_df["band 2"]])
    band_3_test = (band_1_test/band_2_test)
    
    X_test = np.concatenate([band_1_test[:, :, :, np.newaxis], band_2_test[:, :, :, np.newaxis], band_3_test[:, :, :, np.newaxis]], axis=-1)
    
    #returner alt
    return id_array, X_train, y_train, X_test, y_test

def process_img(img):
    new_img = img[:,:,0] + img[:,:,1]
    largest = np.amax(new_img)
    smallest = np.amin(new_img)
    new_img = new_img - smallest
    new_img = new_img * (1/(largest-smallest))
    return new_img

def build_cnn(conv_layers, dense_layers, conv_dropout):
    raw_input = Input(shape=(HEIGHT,WIDTH,CHANNELS))
    normal_input = BatchNormalization()(raw_input)
    layer = Conv2D(conv_layers[0], kernel_size=(3, 3), activation='relu', name="conv0")(normal_input)
    layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)
    if conv_dropout:
        layer = Dropout(conv_dropout)(layer)
    i=1
    for conv_layer_size in conv_layers[1:]:
        layer = Conv2D(conv_layer_size, kernel_size=(3, 3), activation='relu', name="conv"+str(i))(layer)
        layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)
        i+=1
        if conv_dropout:
            layer = Dropout(conv_dropout)(layer)

    layer = Flatten()(layer)

    for dense_layer_size in dense_layers:
        layer = Dense(dense_layer_size, activation='relu')(layer)

    output = Dense(1, activation='sigmoid')(layer)
    optimizer = RMSprop(lr=0.0002)
    model = Model(inputs=raw_input, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def read_model_preds(model_names, regularization, run):
    num_models = len(model_names)
    load_path = MODELS_LOAD_PATH + 'val_preds/run: ' + str(run)
    for model_name in model_names:
        load_path += model_name
    load_path += ", regularization: " + str(regularization) + '.h5'
    df_results = pd.read_hdf(load_path)
    
    preds = np.empty((num_models, df_results.shape[0]))
    for i in range(num_models):
        preds[i] = np.array([pred for pred in df_results['pred'+str(i)]])
    ys = np.array([y for y in df_results['y']])
    
    return preds, ys

def calculate_pairwise_diversity(preds, ys):
    num_pairwise_combos = (len(preds)*(len(preds)-1))/2
    q, rho, disagreement, double_fault = 0,0,0,0
    for idx_pred_1 in range(len(preds)):
        for idx_pred_2 in range(idx_pred_1+1, len(preds)):
            a,b,c,d = 0, 0, 0, 0
            
            for pred_1, pred_2, y in zip(preds[idx_pred_1], preds[idx_pred_2], ys):
                pred_1 = round(pred_1)
                pred_2 = round(pred_2)
                if pred_1 == y:
                    if pred_2 == y:
                        a+=1
                    else:
                        b+=1
                elif pred_2 == y:
                    c+=1
                else: 
                    d+=1
            q += (1/num_pairwise_combos)*((a*d-b*c)/(a*d+b*c))
            #print(idx_pred_1, idx_pred_2, q)
            rho += (1/num_pairwise_combos)*((a*d-b*c)/(np.sqrt((a+b)*(c+d)*(a+c)*(b+d))))
            disagreement += (1/num_pairwise_combos)*((b+c)/(a+b+c+d))
            double_fault += (1/num_pairwise_combos)*(d/(a+b+c+d))
    return q, rho, disagreement, double_fault

def calculate_non_pairwise_diversity(preds, ys):
    num_samples = len(preds[0])
    num_models = len(preds)
    ls = [0]*len(preds[0])
    for pred in preds:
        for idx, (individual_pred, y) in enumerate(zip(pred, ys)):
            if individual_pred == y:
                ls[idx] += 1
    entropy = 0
    for l in ls:
        entropy += (1/num_samples) * (1/(num_models - math.ceil(num_models/2))) * min(l,num_models-l)
    return entropy

def calculate_diversity(preds, ys):
    #diversity calculation
    q, rho, disagreement, double_fault = calculate_pairwise_diversity(preds, ys)
    entropy = calculate_non_pairwise_diversity(preds, ys)
    return q, rho, disagreement, double_fault, entropy

def normalize(x):
    x -= x.min()
    return x/x.max()

def normalize_gradcam(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def grad_cam(model, images, layer_name):
    loss = K.sum(model.output)
    conv_output = [l for l in model.layers if l.name == layer_name][0].output
    grads = normalize_gradcam(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    
    for pic_no, image in enumerate(images):
        output, grads_val = gradient_function([image[np.newaxis,:,:,:]])
        output, grads_val = output[0, :], grads_val[0, :, :, :]

        weights = np.mean(grads_val, axis = (0, 1))
        cam = np.ones(output.shape[0 : 2], dtype = np.float32)
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        cam = cv2.resize(cam, (75, 75))
        cam = np.maximum(cam, 0)

        if pic_no == 0:
            heatmap = np.array(normalize(cam / np.max(cam)))
        else:
            heatmap = np.append(heatmap, np.array(normalize(cam / np.max(cam))))
    
    return heatmap

def run_gradcam_average(regularizations, runs, folds, configs, X_train):
	for regularization in regularizations:
	    heatmap = np.array(0)
	    config = configs[0]
	    layer = "conv" + str(len(config['conv_layers'])-1)
	    model = build_cnn(config['conv_layers'], config['dense_layers'], config['dropout'])
	    for run in range(runs):
	        for fold in range(folds):
	            start = time.time()
	            model_name = 'run: ' + str(run) + ', conv_layers: ' + str(config['conv_layers'])+ ', dense_layers: ' + str(config['dense_layers']) + ', dropout: ' + str(config['dropout']) +'regularization: ' + str(regularization) + ', fold:' + str(fold)
	            print(model_name)
	            model.load_weights(filepath=MODELS_LOAD_PATH + 'weights/' + model_name + '.h5')
	            
	            single_heatmap = grad_cam(model, X_train, layer)
	            
	            plt.hist(x=single_heatmap.flatten(), bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], color='#0504aa',alpha=0.7, rwidth=0.85)
	            plt.tight_layout(pad = 4)
	            plt.grid()
	            plt.title('Average pixel activations for architecture A, regularization ' +str(regularization))
	            plt.ylabel('Count')
	            plt.xlabel('Normalized pixel activation')
	            plt.savefig(MODELS_LOAD_PATH + 'hist_activations/png/all_img_config_'+str(config)+'_regularization_'+str(regularization)+'_run_'+str(run)+'_fold_'+str(fold)+'.png')
	            plt.savefig(MODELS_LOAD_PATH + 'hist_activations/eps/all_img_config_'+str(config)+'_regularization_'+str(regularization)+'_run_'+str(run)+'_fold_'+str(fold)+'.eps', format='eps', dpi=1000)
	            plt.show()
	            plt.clf()
	            
	            heatmap = np.append(heatmap, single_heatmap)
	            
	            end = time.time()
	            print(end-start)

	    plt.hist(x=heatmap.flatten(), bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], color='#0504aa',alpha=0.7, rwidth=0.85)
	    plt.tight_layout(pad = 4)
	    plt.grid()
	    plt.title('Average pixel activations for architecture A, regularization ' +str(regularization), pad=18.0)
	    plt.ylabel('Count')
	    plt.xlabel('Normalized pixel activation')
	    plt.ylim(0, 3.0e08)
	    plt.savefig(MODELS_LOAD_PATH + 'hist_activations/png/all_img_config_'+str(config)+'_regularization_'+str(regularization)+'.png')
	    plt.savefig(MODELS_LOAD_PATH + 'hist_activations/eps/all_img_config_'+str(config)+'_regularization_'+str(regularization)+'.eps', format='eps', dpi=1000)
	    plt.show()
	    plt.clf()
	    
	    hist = np.histogram(heatmap.flatten(), bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
	    with open(MODELS_LOAD_PATH + 'hist_activations/all_img_config_'+str(config)+'_regularization_'+str(regularization)+'.txt', 'w+') as file:
	        file.write(str(hist))

def stacking(num_runs, regularizations, configs):

	model_names = [""]*num_models
	num_regularizations = len(regularizations)
	entropy = np.empty((num_regularizations, num_runs))
	q = np.empty((num_regularizations, num_runs))
	rho = np.empty((num_regularizations, num_runs))
	disagreement = np.empty((num_regularizations, num_runs))
	double_fault = np.empty((num_regularizations, num_runs))
	log_loss_base = np.empty(num_models)
	log_loss_base_avg = np.empty((num_regularizations, num_runs))
	log_loss_combo = np.empty((num_regularizations, num_runs))

	for run in range(num_runs):
	    for i, regularization in enumerate(regularizations):
	        for j, config in enumerate(configs):
	            #name models
	            model_names[j] = 'conv_layers: ' + str(config['conv_layers'])+ ', dense_layers: ' + str(config['dense_layers']) +', dropout: ' + str(config['dropout']) 

	        preds, ys = read_model_preds(model_names, regularization, run)
	        for k, pred in enumerate(preds):
	            log_loss_base[k] = log_loss(ys, pred, eps = PRED_CLIP)
	        log_loss_base_avg[i, run] = np.average(log_loss_base, axis = 0)
	        preds_combo = np.average(preds, axis = 0)
	        log_loss_combo[i, run] = (log_loss(ys, preds_combo, eps = PRED_CLIP))

	        q[i, run], rho[i, run], disagreement[i, run], double_fault[i, run], entropy[i, run] = calculate_diversity(preds, ys)

	q = np.mean(q, axis=1)
	rho = np.mean(rho, axis=1)
	disagreement = np.mean(disagreement, axis=1)
	double_fault = np.mean(double_fault, axis=1)
	entropy = np.mean(entropy, axis=1)
	log_loss_base_avg = np.mean(log_loss_base_avg, axis=1)
	print(log_loss_combo)
	log_loss_combo_avg = np.mean(log_loss_combo, axis=1)
	print(log_loss_combo_avg)
	return q, rho, disagreement, double_fault, entropy, log_loss_base_avg, log_loss_combo_avg, log_loss_combo

def make_plots(num_runs, regularizations, configs):

	q, rho, disagreement, double_fault, entropy, log_loss_base_avg, log_loss_combo_avg, log_loss_combo = stacking(num_runs, regularizations, configs)

	improvement = log_loss_base_avg - log_loss_combo_avg
	
	plt.figure(1)
	plt.plot(q, regularizations, '.')
	plt.grid()
	plt.title('Q-statistic vs Loss improvement')
	plt.ylabel('Loss improvement')
	plt.xlabel('Q-statistic (lower = more diverse)')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/loss_vs_q.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/loss_vs_q.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(2)
	plt.plot(rho, regularizations, '.')
	plt.grid()
	plt.title('Rho vs Loss improvement')
	plt.ylabel('Loss improvement')
	plt.xlabel('Rho (lower = more diverse)')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/loss_vs_rho.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/loss_vs_rho.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(3)
	plt.plot(disagreement, regularizations, '.')
	plt.grid()
	plt.title('Disagreement vs Loss improvement')
	plt.ylabel('Loss improvement')
	plt.xlabel('Disagreement (higher = more diverse)')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/loss_vs_disagreement.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/loss_vs_disagreement.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(4)
	plt.plot(double_fault, regularizations, '.')
	plt.grid()
	plt.title('Double fault vs Loss improvement')
	plt.ylabel('Loss improvement')
	plt.xlabel('Double fault (lower = more diverse)')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/loss_vs_double_fault.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/loss_vs_double_fault.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(5)
	plt.plot(entropy, regularizations, '.')
	plt.grid()
	plt.title('Entropy vs Loss improvement')
	plt.ylabel('Loss improvement')
	plt.xlabel('Entropy (higher = more diverse)')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/loss_vs_entropy.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/loss_vs_entropy.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(6)
	plt.plot(regularizations, log_loss_combo_avg, '.')
	plt.grid()
	plt.title('Diversity emphasis vs ensemble loss')
	plt.ylabel('Ensemble loss ')
	plt.xlabel('Diversity emphasis')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/diversity_emphasis_vs_ensemble_loss.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/diversity_emphasis_vs_ensemble_loss.eps', format='eps', dpi=1000)
	plt.show()
	q = normalize(q)
	double_fault = normalize(double_fault)
	rho = normalize(rho)
	avg_lower = np.mean([q, double_fault, rho], axis = 0)
	plt.figure(8)
	plt.plot(regularizations, avg_lower)
	plt.grid()
	plt.title('Diversity emphasis vs normalized avg value of rho, q and double fault')
	plt.ylabel('Diversity (lower is more)')
	plt.xlabel('Diversity emphasis')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/diversity_emphasis_vs_avg_q_rho_DF.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/diversity_emphasis_vs_avg_q_rho_DF.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(9)
	plt.plot(regularizations, q)
	plt.grid()
	plt.title('Diversity emphasis vs normalized q')
	plt.ylabel('Q (lower is more)')
	plt.xlabel('Diversity emphasis')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/diversity_emphasis_vs_q.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/diversity_emphasis_vs_q.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(10)
	plt.plot(regularizations, rho)
	plt.grid()
	plt.title('Diversity emphasis vs normalized rho')
	plt.ylabel('Rho (lower is more)')
	plt.xlabel('Diversity emphasis')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/diversity_emphasis_vs_avg_rho.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/diversity_emphasis_vs_avg_rho.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(11)
	plt.plot(regularizations, double_fault)
	plt.grid()
	plt.title('Diversity emphasis vs normalized double fault')
	plt.ylabel('Double fault (lower is more)')
	plt.xlabel('Diversity emphasis')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/diversity_emphasis_vs_DF.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/diversity_emphasis_vs_DF.eps', format='eps', dpi=1000)
	plt.show()
	entropy = normalize(entropy)
	disagreement = normalize(disagreement)
	avg_higher = np.mean([entropy, disagreement], axis = 0)
	plt.figure(12)
	plt.plot(regularizations, avg_higher)
	plt.grid()
	plt.title('Diversity emphasis vs normalized avg value of disagreement and entropy')
	plt.ylabel('Diversity (higher is more)')
	plt.xlabel('Diversity emphasis')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/diversity_emphasis_vs_avg_entropy_disagreement.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/diversity_emphasis_vs_avg_entropy_disagreement.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(13)
	plt.plot(regularizations, entropy)
	plt.grid()
	plt.title('Diversity emphasis vs normalized entropy')
	plt.ylabel('Entropy (higher is more)')
	plt.xlabel('Diversity emphasis')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/diversity_emphasis_vs_entropy.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/diversity_emphasis_vs_entropy.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(14)
	plt.plot(regularizations, disagreement)
	plt.grid()
	plt.title('Diversity emphasis vs normalized disagreement')
	plt.ylabel('Disagreement (higher is more)')
	plt.xlabel('Diversity emphasis')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/diversity_emphasis_vs_disagreement.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/diversity_emphasis_vs_disagreement.eps', format='eps', dpi=1000)
	plt.show()
	plt.figure(15)
	plt.plot(regularizations, log_loss_combo[:,1], '.')
	plt.grid()
	plt.title('Diversity emphasis vs ensemble loss individual runs')
	plt.ylabel('Loss')
	plt.xlabel('Diversity emphasis')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/png/diversity_emphasis_vs_ensemble_loss_individual_runs.png')
	plt.savefig(MODELS_LOAD_PATH + 'div_vs_loss/eps/diversity_emphasis_vs_ensemble_loss_individual_runs.eps', format='eps', dpi=1000)
	plt.show()


	df_div_vs_loss = pd.DataFrame({'log_loss_base_avg' : log_loss_base_avg, 'log_loss_combo' : log_loss_combo_avg, 
	                               'improvement' : log_loss_base_avg - log_loss_combo_avg,  'q' : q, 'rho' : rho,  
	                               'disagreement' : disagreement, 'double_fault' : double_fault,  'entropy' : entropy}, index = regularizations)
	df_div_vs_loss.to_csv(MODELS_LOAD_PATH + 'div_vs_loss/div_vs_loss.csv')
        


MODELS_LOAD_TIME = "2019-03-27 10:56:00.344909"
MODELS_LOAD_PATH = './ncl/' + MODELS_LOAD_TIME + '/'
PRED_CLIP = 0.00001
HEIGHT = 75
WIDTH = 75
CHANNELS = 3
configs = []
regularizations = [0.0, 1e-13, 1e-12, 1e-11]
num_folds = 4
num_models = 12
num_runs = 4

conv_layers_to_try = [
    [64, 128, 256],
    [32, 64, 128, 256]
]
dense_layers_to_try = [
    #[128, 64],
    [256, 128],
    [512, 256]
]
dropout_to_try = [0, 0.2]

configs = []
for conv_layers in conv_layers_to_try:
    for dense_layers in dense_layers_to_try:
        for dropout in dropout_to_try:
            config = dict(conv_layers = conv_layers,
                          dense_layers = dense_layers,
                          dropout = dropout)
            configs.append(config)

#import data
id_array, X_train, y_train, X_test, y_test = load_equinor(0.2)

run_gradcam_average(regularizations, runs, folds, configs, X_train)
#make_plots(num_runs, regularizations, configs)