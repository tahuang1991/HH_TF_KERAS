import glob, os, re, math, socket, json, array
import numpy as np
import pandas as pd
import numpy.lib.recfunctions as recfunc
from   ROOT import TChain, TFile, TTree, TH1F
from   root_numpy import tree2array
from   sklearn.model_selection import train_test_split
from   sklearn.utils import shuffle, safe_indexing
import tensorflow as tf
import keras
from   keras.callbacks import Callback
import matplotlib.pyplot as plt
import keras.backend as K
from   matplotlib.collections import PatchCollection
from   matplotlib.patches import Rectangle
import configuration as conf

def tree_to_numpy(input_file, variables, weight, cut=None, reweight_to_cross_section=False):
    file_handle = TFile.Open(input_file)
    tree = file_handle.Get('evtreeHME')
    cross_section = 1
    relative_weight = 1
    if reweight_to_cross_section:
        h_xSec = TH1F("h_xSec","",10000,0.,10000); tree.Draw("cross_section>>h_xSec","","nog")
        cross_section = h_xSec.GetMean()
        if (h_xSec.GetRMS()/h_xSec.GetMean()>0.0001):
          print "WARNING: cross_section has not a single value!!! RMS is", h_xSec.GetRMS()
        h_weiSum = TH1F("h_weiSum","",1000000000,0.,1000000000); tree.Draw("event_weight_sum>>h_weiSum","","nog")
        if (h_weiSum.GetRMS()/h_weiSum.GetMean()>0.0001):
          print "WARNING: event_weight_sum has not a single value!!! RMS is", h_weiSum.GetRMS()
        relative_weight = cross_section / h_weiSum.GetMean()
    if isinstance(weight, dict): # Returns a Boolean stating whether the object is an instance or subclass of another object.
        # Keys are regular expression and values are actual weights. Find the key matching the input filename
        found, weight_expr = False, None
        if '__base__' in weight:
            weight_expr = weight['__base__']
        for k, v in weight.items():
            if k == '__base__':
                continue
            groups = re.search(k, input_file)
            if not groups:
                continue
            else:
                if found:
                    raise Exception("The input file is matched by more than one weight regular expression. %r" % input_file)
                found = True
                weight_expr = join_expression(weight_expr, v)
        if not weight_expr:
            raise Exception("Not weight expression found for input file %r" % weight_expr)
        weight = weight_expr
    # Read the tree and convert it to a numpy structured array
    a = tree2array(tree, branches=variables + [weight], selection=cut)
    # Rename the last column to 'weight'
    a.dtype.names = variables + ['weight']
    dataset = a[variables]
    weights = a['weight'] * relative_weight
    dataset = np.array(dataset.tolist(), dtype=np.float32)
    return dataset, weights

def save_training_parameters(output, model, **kwargs):
    parameters = {
            'extra': kwargs
            }

    model_definition = model.to_json()
    m = json.loads(model_definition)
    parameters['model'] = m

    with open(os.path.join(output, 'parameters.json'), 'w') as f:
        json.dump(parameters, f, indent=4)

def binDataset(dataset, weights, bins, range=None):
    """
    Bin a dataset
    Parameters:
        dataset: a numpy array of data to bin
        weights: data weights
        bins: either a list of bin boundaries or the number of bin to user
        range: The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()). Values outside the range are ignored
    Returns:
        a tuple (hist, errors, bin_edges):
    """
    # First, bin dataset
    hist, bin_edges = np.histogram(dataset, bins=bins, range=range, weights=weights)
    # Bin weights^2 to extract the uncertainty
    squared_errors, _ = np.histogram(dataset, weights=np.square(weights), bins=bin_edges)
    errors = np.sqrt(squared_errors)
    norm_factor = np.diff(bin_edges) * hist.sum()
    hist /= norm_factor
    errors /= norm_factor

    return (hist, errors, bin_edges)

def makeErrorBoxes(ax, xdata, ydata, yerror, bin_width, ec='None', alpha=0.2):
    # Create list for all the error patches
    errorboxes = []
    # Loop over data points; create box from errors at each point
    for xc, yc, ye in zip(xdata, ydata, yerror):
        rect = Rectangle((xc - bin_width / 2, yc - ye), bin_width, 2*ye)
        errorboxes.append(rect)
    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, linewidth=0, facecolor='none', edgecolor=ec, hatch='/////', alpha=alpha)
    # Add collection to axes
    ax.add_collection(pc)

def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * exp(-k*t)
   return lrate

def lr_scheduler(epoch): # a function that takes an epoch index as input and provide learning rate
    default_lr = 0.001
    drop = 0.1
    epochs_drop = 50.0
    lr = default_lr * math.pow(drop, min(1, math.floor((1 + epoch) / epochs_drop)))
    return lr

# Sensitivity, true positive rate, recall: proportion of actual positives that are correctly identified as such
def sensitivity(y_true, y_pred):
    true    = tf.cast(tf.constant([[1,0]]), tf.float32)
    y_true  = tf.keras.backend.round(tf.cast(y_true, tf.float32))
    y_pred  = tf.keras.backend.round(tf.cast(y_pred, tf.float32))
    #Predicted positive that are actually positive
    positive_found = tf.divide(tf.count_nonzero(tf.multiply( tf.cast(tf.equal(y_true, y_pred),tf.int32), tf.cast(tf.equal(y_true,true),tf.int32))), true.get_shape()[1])
    #All positives
    all_positives  = tf.divide(tf.count_nonzero(tf.equal(y_true, true)), true.get_shape()[1])
    return positive_found / (all_positives + tf.keras.backend.epsilon())

# Specificity, true negative rate: proportion of actual negatives that are correctly identified as such
def specificity(y_true, y_pred): # Background_identified/Real_Background
    false   = tf.cast(tf.constant([[0,1]]), tf.float32)
    y_true  = tf.keras.backend.round(tf.cast(y_true, tf.float32))
    y_pred  = tf.keras.backend.round(tf.cast(y_pred, tf.float32))
    #Predicted negative that are actually negative
    negatives_found = tf.divide(tf.count_nonzero(tf.multiply( tf.cast(tf.equal(y_true, y_pred),tf.int32), tf.cast(tf.equal(y_true,false),tf.int32))), false.get_shape()[1])
    #All negative
    all_negatives   = tf.divide(tf.count_nonzero(tf.equal(y_true, false)), false.get_shape()[1])
    return negatives_found / (all_negatives + tf.keras.backend.epsilon())

# Step decay for LR
def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

# Record loss history and learning rate during the training procedure
class LossHistory(keras.callbacks.Callback):
    sess = tf.InteractiveSession()
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append((self.model.optimizer.lr).eval())
    def save_to_py(self, folder):
        np.save(folder + '/call_losses.npy', self.losses)
        np.save(folder + '/call_lr.npy', self.lr)
        self.losses
        self.lr

# https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/
class LRFinder(Callback):
#! class LRFinder(tf.keras.callbacks):
#class LRFinder(keras.callbacks):
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            lr_finder.plot_loss()
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super(LRFinder, self).__init__() #super() lets you avoid referring to the base class explicitly. The main advantage comes with multiple inheritance.
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        print 'AAAAAA on_train_begin', self.model.optimizer.lr, self.min_lr
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        print 'on_train_begin'
#        
#    def on_batch_end(self, epoch, logs=None):
#        '''Record previous batch statistics and update the learning rate.'''
#        print 'AAAAAA on_batch_end'
#        logs = logs or {}
#        self.iteration += 1
#        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
#        self.history.setdefault('iterations', []).append(self.iteration)
#        for k, v in logs.items():
#            self.history.setdefault(k, []).append(v)
#        K.set_value(self.model.optimizer.lr, self.clr())
#        print 'on_batch_end'
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')

class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)
