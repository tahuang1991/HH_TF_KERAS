import os
import sys
import gzip
import math
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from   keras.regularizers import l2
from   keras.utils.vis_utils import plot_model
from   sklearn.utils import class_weight
import common as com
import configuration as conf
'''
Goal: optimize hyperparamters.
'''
customized_name = 'test1' 
# Parameters
data_folder     = conf.data_folder
mass            = sys.argv[1]
layers          = int(sys.argv[2])
nodes           = int(sys.argv[3])
lr              = float(sys.argv[4])
batch_size      = int(sys.argv[5])
customized_name = 'opt_layers' + str(layers) + '_nodes' + str(nodes) + '_lr' + str(lr) + '_batch' + str(batch_size)
print '---- MASS', mass, '----'
df_train        = pd.read_csv(data_folder + '/train_dataset' + str(mass) + '.csv')
df_train_weight = pd.read_csv(data_folder + '/train_weights' + str(mass) + '.csv')
df_train_target = pd.read_csv(data_folder + '/train_targets' + str(mass) + '.csv')
df_valid        = pd.read_csv(data_folder + '/valid_dataset' + str(mass) + '.csv')
df_valid_weight = pd.read_csv(data_folder + '/valid_weights' + str(mass) + '.csv')
df_valid_target = pd.read_csv(data_folder + '/valid_targets' + str(mass) + '.csv')
df_test         = pd.read_csv(data_folder + '/test_dataset' + str(mass) + '.csv')
df_test_weight  = pd.read_csv(data_folder + '/test_weights' + str(mass) + '.csv')
df_test_target  = pd.read_csv(data_folder + '/test_targets' + str(mass) + '.csv')
print 'Training on', df_train_target.loc[df_train_target['is_s']==1].shape[0], 'Signal, and',  df_train_target.loc[df_train_target['is_b']==1].shape[0], 'Background.'
folder_name     = conf.base_folder + str(conf.training_epochs) + conf.output_suffix + str(mass) + '_' + customized_name + '/'
model_name      = conf.model_name
model_NamePath  = folder_name + '/' + model_name
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Class weights
weight_class_s = 1./(float(df_train_weight.loc[df_train_target['is_s']==1].shape[0]) / df_train_weight.shape[0])
weight_class_b = 1./(float(df_train_weight.loc[df_train_target['is_s']==0].shape[0]) / df_train_weight.shape[0])
print '#S times S_Weight = ', df_train_weight.loc[df_train_target['is_s']==1].shape[0], '*', weight_class_s, '=', df_train_weight.loc[df_train_target['is_s']==1].shape[0]*weight_class_s 
print '#B times B_Weight = ', df_train_weight.loc[df_train_target['is_s']==0].shape[0], '*', weight_class_b, '=', df_train_weight.loc[df_train_target['is_s']==0].shape[0]*weight_class_b
df_train_weight['weight_class'] = weight_class_s # All has S weight
df_train_weight['weight_class'] = np.where(df_train_target['is_s']==0, weight_class_b, df_train_weight.weight_class) # where(condition, 'yes', 'no')
df_valid_weight['weight_class'] = weight_class_s
df_valid_weight['weight_class'] = np.where(df_valid_target['is_s']==0, weight_class_b, df_valid_weight.weight_class)
df_test_weight['weight_class']  = weight_class_s
df_test_weight['weight_class']  = np.where(df_test_target['is_s']==0, weight_class_b, df_test_weight.weight_class)

# Create model
n_inputs = len(conf.features)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(nodes, kernel_initializer="glorot_uniform", activation="relu", input_dim=n_inputs, kernel_regularizer=conf.regular)) 
for nL in range(layers-1):
    model.add(tf.keras.layers.Dense(nodes, kernel_initializer="glorot_uniform", activation="relu"))
model.add(tf.keras.layers.Dense(2, kernel_initializer="glorot_uniform", activation="softmax"))
# Compile the model (before training a model, you need to configure the learning process)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=lr), metrics=['accuracy', com.sensitivity, com.specificity])
model.summary()
#! plot_model(model, to_file=folder_name+'/model_plot.png', show_shapes=True, show_layer_names=True)
# Callback: set of functions to be applied at given stages of the training procedure
callbacks = []
callbacks.append(tf.keras.callbacks.ModelCheckpoint(model_NamePath, monitor='val_loss', verbose=False, save_best_only=True))
#!callbacks.append(tf.keras.callbacks.LearningRateScheduler(com.lr_scheduler)) # How this is different from 'conf.optimiz' ?
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5)) # Stop once you stop improving the val_loss
# Train the model
history = model.fit(df_train[conf.features],
                    df_train_target[['is_s','is_b']],
                    sample_weight   = df_train_weight['weight_class'],
                    batch_size      = batch_size,
                    epochs          = conf.training_epochs,
                    validation_data = (df_test[conf.features], df_test_target[['is_s','is_b']], df_test_weight['weight_class']),
                    callbacks       = callbacks)

com.save_training_parameters(folder_name, model,
          batch_size=batch_size, epochs=conf.training_epochs,
          masses=mass,
          with_mass_column=False,
          inputs=conf.features,
          cut=conf.selection,
          weights=conf.resonant_weights)

# Plot loss value for training and validation samples
print 'Saving loss vs epochs.'
training_losses = history.history['loss']
validation_losses = history.history['val_loss']
epochs = np.arange(0, len(training_losses))
l1 = plt.plot(epochs, training_losses, '-', color='#8E2800', lw=2, label="Training loss")
l2 = plt.plot(epochs, validation_losses, '-', color='#468966', lw=2, label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
lns = l1 + l2
labs = [l.get_label() for l in lns]
plt.legend(lns, labs, loc='best', numpoints=1, frameon=False)
plt.savefig(folder_name + '/loss.png')
plt.clf()

# Saving the results
print 'Saving history.'
json.dump(history.history, open(folder_name + '/model_history.json', 'w'))
print 'Saving Model'
tf.keras.models.save_model(model, folder_name+'/my_model.h5', overwrite=True)
