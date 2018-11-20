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
Goal: train the learner and save the model.
'''
# Parameters
data_folder     = conf.data_folder
mass            = sys.argv[1]
print '---- MASS', mass, '----'
Extra_label     = '_PCA' if conf.usePCA else ''
df_train        = pd.read_csv(data_folder + '/train_dataset' + str(mass) + Extra_label + '.csv')
df_train_weight = pd.read_csv(data_folder + '/train_weights' + str(mass) + '.csv')
df_train_target = pd.read_csv(data_folder + '/train_targets' + str(mass) + '.csv')
df_valid        = pd.read_csv(data_folder + '/valid_dataset' + str(mass) + Extra_label + '.csv')
df_valid_weight = pd.read_csv(data_folder + '/valid_weights' + str(mass) + '.csv')
df_valid_target = pd.read_csv(data_folder + '/valid_targets' + str(mass) + '.csv')
df_test         = pd.read_csv(data_folder + '/test_dataset' + str(mass) + Extra_label + '.csv')
df_test_weight  = pd.read_csv(data_folder + '/test_weights' + str(mass) + '.csv')
df_test_target  = pd.read_csv(data_folder + '/test_targets' + str(mass) + '.csv')
print 'Training on', df_train_target.loc[df_train_target['is_s']==1].shape[0], 'Signal, and',  df_train_target.loc[df_train_target['is_b']==1].shape[0], 'Background.'
folder_name       = conf.base_folder + str(conf.training_epochs) + conf.output_suffix + str(mass)
featureReduced    = len(conf.features) - len(conf.features_PCA)
folder_name       = folder_name + '_PCAm' + str(featureReduced) + '/' if conf.usePCA else folder_name + '/'
model_name        = conf.model_name
model_NamePath    = folder_name + '/' + model_name
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
features = conf.features_PCA if conf.usePCA else conf.features
n_inputs = len(conf.features_PCA) if conf.usePCA else len(conf.features)

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
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(60, kernel_initializer="glorot_uniform", activation="relu", input_dim=n_inputs, kernel_regularizer=conf.regular)) 
model.add(tf.keras.layers.Dense(60, kernel_initializer="glorot_uniform", activation="relu"))
model.add(tf.keras.layers.Dense(2, kernel_initializer="glorot_uniform", activation="softmax"))

# Compile the model (before training a model, you need to configure the learning process)
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=conf.lr), metrics=[com.sensitivity, com.specificity])
model.summary()
# Callback: set of functions to be applied at given stages of the training procedure
callbacks = []
callbacks.append(tf.keras.callbacks.ModelCheckpoint(model_NamePath, monitor='val_loss', verbose=False, save_best_only=True))
callbacks.append(tf.keras.callbacks.LearningRateScheduler(com.lr_scheduler)) # Overwrite lr in the Optimizer, i.e. Adam.
callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=7)) # Stop once you stop improving the val_loss
# Append to callbacks to find the the best lr. I would also remove teh early stopping since you need to know the number of epochs.
loss_history = com.LossHistory()
callbacks.append(loss_history)
#lr_finder = com.LRFinder(min_lr=1e-5,
#                         max_lr=1e-2,
#                         steps_per_epoch = np.ceil(df_train_target.shape[0]/conf.batch_size),
#                         epochs=conf.training_epochs)
#! callbacks.append(lr_finder)

# Train the model
history = model.fit(df_train[features],
                    df_train_target[['is_s','is_b']],
                    sample_weight   = df_train_weight['weight_class'],
                    batch_size      = conf.batch_size,
                    epochs          = conf.training_epochs,
                    validation_data = (df_valid[features], df_valid_target[['is_s','is_b']], df_valid_weight['weight_class']),
                    callbacks       = callbacks)

com.save_training_parameters(folder_name, model,
          batch_size=conf.batch_size, epochs=conf.training_epochs,
          masses=mass,
          with_mass_column=False,
          inputs=features,
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

# Save LR and Loss history
numpy_folder = conf.numpy_folder
if not os.path.exists(folder_name + '/' + numpy_folder):
    os.makedirs(folder_name + '/' + numpy_folder)
loss_history.save_to_py(folder_name + '/' + numpy_folder)

# Saving the results
print 'Saving history.'
json.dump(history.history, open(folder_name + '/model_history.json', 'w'))
print 'Saving Model'
tf.keras.models.save_model(model, folder_name+'/my_model.h5', overwrite=True)
