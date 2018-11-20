import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import keras
from   sklearn.metrics import confusion_matrix
from   sklearn.metrics import roc_curve
from   sklearn.metrics import auc
import common as com
import configuration as conf
'''
Goal: quantiy the performance of a model.
'''
# Parameters
data_folder = conf.data_folder
mass        = sys.argv[1]
folder_name = sys.argv[2]
print '---- Folder', folder_name, '----'
PCA_label       = '_PCA' if conf.usePCA else ''
df_train        = pd.read_csv(data_folder + '/train_dataset' + str(mass) + PCA_label + '.csv')
df_train_weight = pd.read_csv(data_folder + '/train_weights' + str(mass) + '.csv')
df_train_target = pd.read_csv(data_folder + '/train_targets' + str(mass) + '.csv')
df_valid        = pd.read_csv(data_folder + '/valid_dataset' + str(mass) + PCA_label + '.csv')
df_valid_weight = pd.read_csv(data_folder + '/valid_weights' + str(mass) + '.csv')
df_valid_target = pd.read_csv(data_folder + '/valid_targets' + str(mass) + '.csv')
df_test         = pd.read_csv(data_folder + '/test_dataset' + str(mass) + PCA_label + '.csv')
df_test_weight  = pd.read_csv(data_folder + '/test_weights' + str(mass) + '.csv')
df_test_target  = pd.read_csv(data_folder + '/test_targets' + str(mass) + '.csv')
features = conf.features_PCA if conf.usePCA else conf.features

print 'Evaluating performance of', folder_name
print '(Trained on', df_train_target.loc[df_train_target['is_s']==1].shape[0], 'Signal, and',  df_train_target.loc[df_train_target['is_b']==1].shape[0], 'Background).'
loaded_model = tf.keras.models.load_model(folder_name + '/my_model.h5', custom_objects={'sensitivity':com.sensitivity, 'specificity':com.specificity})

# Evaluate loaded model on Train/Test data
score_train = loaded_model.evaluate(df_train[features], df_train_target[['is_s','is_b']], verbose=0)
score_test  = loaded_model.evaluate(df_test[features], df_test_target[['is_s','is_b']], verbose=0)
for i, matric_name in enumerate(loaded_model.metrics_names):
    print("Train and Test score for %s: %.2f%% and %.2f%%" % (matric_name, score_train[i]*100., score_test[i]*100))

# Plot loss value for training and validation samples
with open(folder_name + '/model_history.json') as f:
    history = json.load(f)
for metric in history:
    if 'val_' not in metric:
        training_metric   = history[metric]
        validation_metric = history['val_'+metric]
        epochs            = np.arange(0, len(training_metric))
        l1 = plt.plot(epochs, training_metric, '-', color='#8E2800', lw=2, label="Training " + metric)
        l2 = plt.plot(epochs, validation_metric, '-', color='#468966', lw=2, label="Validation " + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc='best', numpoints=1, frameon=False)
        plt.savefig(folder_name + '/metric_' + metric + '.png')
        plt.clf()

# LR, Loss etc...
loss = np.load(folder_name + '/numpy/call_losses.npy')
lr   = np.load(folder_name + '/numpy/call_lr.npy')
plt.plot(lr, loss, label='lr vs loss', color='blue')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('LR vs Loss')
plt.savefig(folder_name+'/LR_loss.png')
plt.clf()
plt.plot(range(len(lr)), lr, label='Epochs vs lr', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Epochs vs LR')
plt.savefig(folder_name+'/LR.png')
plt.clf()

# DNN output
fig = plt.figure(1, figsize=(7, 7), dpi=300)
ax = fig.add_subplot(111)
# Training data
training_signal_data        = df_train[features].loc[df_train_target['is_s']==1]
training_signal_weights     = df_train_weight['weight'].loc[df_train_target['is_s']==1]
training_background_data    = df_train[features].loc[df_train_target['is_b']==1]
training_background_weights = df_train_weight['weight'].loc[df_train_target['is_b']==1]
# Testing
testing_signal_data         = df_test[features].loc[df_test_target['is_s']==1]
testing_signal_weights      = df_test_weight['weight'].loc[df_test_target['is_s']==1]
testing_background_data     = df_test[features].loc[df_test_target['is_b']==1]
testing_background_weights  = df_test_weight['weight'].loc[df_test_target['is_b']==1]
# Predictions. Predic returns [prob_s, prob_b]. Deleting column 1 you get [prob_s]
training_signal_predictions     = np.delete(loaded_model.predict(training_signal_data, batch_size=5000, verbose=1), 1, axis=1).flatten()
testing_signal_predictions      = np.delete(loaded_model.predict(testing_signal_data, batch_size=5000, verbose=1), 1, axis=1).flatten()
training_background_predictions = np.delete(loaded_model.predict(training_background_data, batch_size=5000, verbose=1), 1, axis=1).flatten()
testing_background_predictions  = np.delete(loaded_model.predict(testing_background_data, batch_size=5000, verbose=1), 1, axis=1).flatten()
# Histogram with the 'Prob_S' for S and B. Uncertainty also is returned as sqrt(total_weight).
training_background_histogram, training_background_errors, bin_edges = com.binDataset(training_background_predictions, training_background_weights, bins=50, range=[0,1])
training_signal_histogram, training_signal_errors, _ = com.binDataset(training_signal_predictions, training_signal_weights, bins=bin_edges)
testing_background_histogram, testing_background_errors, _ = com.binDataset(testing_background_predictions, testing_background_weights, bins=bin_edges)
testing_signal_histogram, testing_signal_errors, _ = com.binDataset(testing_signal_predictions, testing_signal_weights, bins=bin_edges)
# Plot it correctly
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_width = bin_edges[1] - bin_edges[0]
ax.bar(bin_centers, training_background_histogram, lw=0, align='center', alpha=0.5, label='Background (training)', color=conf.background_color, width=bin_width)
com.makeErrorBoxes(ax, bin_centers, training_background_histogram, training_background_errors, bin_width, ec=conf.background_color, alpha=0.7)
ax.bar(bin_centers, training_signal_histogram, lw=0, align='center', alpha=0.5, label='Signal (training)', color=conf.signal_color, width=bin_width)
com.makeErrorBoxes(ax, bin_centers, training_signal_histogram, training_signal_errors, bin_width, ec=conf.signal_color, alpha=0.7)
ax.errorbar(bin_centers, testing_background_histogram, yerr=testing_background_errors, linestyle='', marker='o', mew=0, mfc=conf.background_color, ecolor=conf.background_color, label='Background (testing)')
ax.errorbar(bin_centers, testing_signal_histogram, yerr=testing_signal_errors, linestyle='', marker='o', mew=0, mfc=conf.signal_color, ecolor=conf.signal_color, label='Signal (testing)')
ax.margins(x=0.1)
ax.set_ylim(ymin=0)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], ncol=2, numpoints=1, loc='best', frameon=False)
ax.set_xlabel("NN output")
fig.set_tight_layout(True)
fig.savefig(folder_name + '/DNN_output.png')
plt.close()
 
# ROC Curve. From ['Promb_S','Prob_B'] I need a single value: 0=Sig, 1=Bkg.
pred_is_sig_binary = np.argmin(loaded_model.predict(df_test[features], batch_size=5000, verbose=1), axis=1) # argmin is the index of the min value. [0.6,0.4] = sig = 1, while [0.3,0.7] = bkg = 0
pred_is_sig        = loaded_model.predict(df_test[features], batch_size=5000, verbose=1)[:,0] # Prob_s
test_is_sig        = np.argmin(df_test_target[['is_s','is_b']].values, axis=1) # argmin is the index of the min value. [1,0] = sig = 1, while [0,1] = bkg = 0

print 'Confusion Matrix:'
print confusion_matrix(test_is_sig, pred_is_sig_binary)
# ROC curve
numpy_folder = conf.numpy_folder
if not os.path.exists(folder_name + '/' + numpy_folder):
    os.makedirs(folder_name + '/' + numpy_folder)
fpr, tpr, _ = roc_curve( test_is_sig, pred_is_sig ) # It returns: fpr, tpr, thresholds
np.save(folder_name + '/' + numpy_folder + '/fpr.npy', fpr)
np.save(folder_name + '/' + numpy_folder + '/tpr.npy', tpr)
auc_keras = auc(fpr, tpr)
# Now plot ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(folder_name+'/HhhKeras_ROC.png')
plt.clf()
print 'DONE!'
