import ROOT
import numpy as np
import pickle
import math
import datetime
import json
from timeit import default_timer as timer
import gzip


# Select Theano as backend for Keras
#from os import environ
import os
os.environ['KERAS_BACKEND'] = 'theano'

# Set architecture of system (AVX instruction set is not supported on SWAN)
#environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, safe_indexing



import configuration as conf
import plotTools
from helper import *


epochs = 50
batch_size = 1000
dataLumi = 35.86

treename = "t"
sfile_dict = {}
for m in [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 750, 800, 900]:
    sfile_dict["RadiaonM%d"%m] = {}
    sfile_dict["RadiaonM%d"%m]['path'] = "/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/radion_M%d_all.root"%m
    sfile_dict["RadiaonM%d"%m]['weight'] = "total_weight"


bfile_dict = {
        "TT" : {
            'path':"/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/TT_all.root",
            'weight': "total_weight"
            },
        "DYM10to50" : {
            'path':"/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/DYM10to50_all.root",
            'weight': "total_weight*dy_nobtag_to_btagM_weight"
            },
        "DYToLL0J" : {
            'path':"/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/DYToLL0J_all.root",
            'weight': "total_weight*dy_nobtag_to_btagM_weight"
            },
        "DYToLL1J" : {
            'path':"/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/DYToLL1J_all.root",
            'weight': "total_weight*dy_nobtag_to_btagM_weight"
            },
        "DYToLL2J" : {
            'path':"/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/DYToLL2J_all.root",
            'weight': "total_weight*dy_nobtag_to_btagM_weight"
            },
        "sTop" : {
            'path':"/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/sT_top_all.root",
            'weight': "total_weight"
            },
        "santiTop" : {
            'path':"/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/sT_antitop_all.root",
            'weight': "total_weight"
            }
        }


def create_resonant_model(n_inputs):
    # Define the model
    model = Sequential()
    # kernel_initializer: Initializations define the way to set the initial random weights of Keras layers (glorot_uniform = uniform initializer)
    # activation: activation function for the nodes. relu is good for intermedied step, softmax for the last one.
    model.add(Dense(100, kernel_initializer="glorot_uniform", activation="relu", input_dim=n_inputs))
    n_hidden_layers = 4
    for i in range(n_hidden_layers):
        model.add(Dense(100, kernel_initializer="glorot_uniform", activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, kernel_initializer="glorot_uniform", activation="softmax"))
    #optimizer = Adam(lr=0.0001)
    optimizer = Adam(lr=0.0005)
    # You compile it so you can actually use it
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

 
class DatasetManager:
    def __init__(self, variables, selection, masses):
        """
        Create a new dataset manager

        Parameters:
            variables: list of input variables. This can be either branch names or a more
              complexe mathematical expression
            selection: a cut expression applied to each event. Only events passing this selection are
              kept
        """

        self.variables = variables
        self.selection = selection
        self.resonant_masses = masses

       ## only for parametric DNN 
        self.resonant_mass_probabilities = None
        self.parametricDNN = False
        if len(masses) > 1:     self.parametricDNN = True

        print("build dataloader object, with signal mass ", masses, " selections ",selection, " input variables ", variables)
        print("\n")


    def loadResonantSignal(self, treename):
        masses = self.resonant_masses
        print("loading Resonant Signals.....")
        p = [[], []]
        datasets = []
        weights = []


        for m in masses:
            signal_filename = sfile_dict["RadiaonM%d"%m]['path']
            weightexpr = sfile_dict["RadiaonM%d"%m]['weight']
            df_sig = ROOT.RDataFrame(treename, signal_filename)
            ## convert df_sig into dict
            ## isSF is boolean and should be treated differently 
            data_sig = df_sig.Filter(self.selection).Define('final_total_weight', weightexpr).Define('isSF_float','(1.0*isSF)').AsNumpy() 
            #print("isSF ", data_sig['isSF'],' isSF_float ',data_sig['isSF_float'])
            self.variables = map(lambda x : x if x != 'isSF' else 'isSF_float', self.variables)
            #for i in range(50):
            #    print("jj_M: ", data_sig['jj_M'][i], " isSF ",data_sig['isSF'][i],' isSF_float  ', data_sig['isSF_float'][i]  )
            
            ## convert into ndarray
            x_sig = np.vstack([data_sig[var] for var in self.variables]).T
            w_sig = data_sig['final_total_weight']
            #x_sig = np.array(x_sig.tolist(), dtype=np.float32)

            p[0].append(m)
            p[1].append(len(x_sig))

            if self.parametricDNN:
                mass_col = np.empty(len(x_sig)) 
                mass_col.fill(m)
                x_sig = np.c_[x_sig, mass_col]

            datasets.append(x_sig)
            weights.append(w_sig)


        # Normalize probabilities in order that sum(p) = 1
        p[1] = np.array(p[1], dtype='float')
        p[1] /= np.sum(p[1])

        self.signal_dataset = np.concatenate(datasets)
        self.signal_weights = np.concatenate(weights)
        self.resonant_mass_probabilities = p

        print("Done. Number of signal events: %d ; Sum of weights: %.4f" % (len(self.signal_dataset), np.sum(self.signal_weights)))
        print("\n")
        #return datasets,weights

    def loadBackgrounds(self, treename):
        print("loading backgrounds.....")
        datasets = []
        weights = []

        for bg in bfile_dict.keys():
        #for bg in ['sTop']:
            bg_filename = bfile_dict[bg]['path']
            weightexpr = bfile_dict[bg]['weight']
            xsec,event_weight_sum = get_xsection_eventweightsum_file(bg_filename, treename)
            relative_weight = dataLumi*1000*xsec/event_weight_sum
            df_bg = ROOT.RDataFrame(treename, bg_filename)
            data_bg = df_bg.Filter(self.selection).Define('final_total_weight', weightexpr+"*%f"%(relative_weight)).Define('isSF_float','(1.0*isSF)').AsNumpy()
            x_bg = np.vstack([data_bg[var] for var in self.variables]).T
            x_bg = np.array(x_bg.tolist(), dtype=np.float32)
            w_bg = data_bg['final_total_weight']

            ### add mass column for bg, paramtric DNN
            if self.parametricDNN and self.resonant_mass_probabilities != None:
                probabilities = self.resonant_mass_probabilities
                rs = np.random.RandomState(42)

                indices = rs.choice(len(probabilities[0]), len(x_bg), p=probabilities[1])
                cols = np.array(np.take(probabilities[0], indices, axis=0), dtype='float')
                x_bg = np.c_[x_bg, cols]
            elif self.parametricDNN:
                print("Warning!!! it is a parametric DNN training but no extra column is added")

            print("loading background ",bg, " number of events: %d ;  final yield: %4.f" % (len(x_bg), np.sum(w_bg)))
            datasets.append(x_bg)
            weights.append(w_bg)

        self.background_dataset = np.concatenate(datasets)
        self.background_weights = np.concatenate(weights)

        print("Done. Number of background events: %d ; final yield: %.4f " % (len(self.background_dataset), np.sum(self.background_weights)))
        print("\n")

        #return datasets, weights

    def update_background_mass_column(self):
        rs = np.random.RandomState(42)
        mass_col = np.array(rs.choice(self.resonant_mass_probabilities[0], len(self.background_dataset), p=self.resonant_mass_probabilities[1]), dtype='float')

        self.background_dataset[:, len(self.variables)] = mass_col


    def split(self, reweight_background_training_sample=True):
        """
        Split datasets into a training and testing samples

        Parameter:
            reweight_background_training_sample: If true, the background training sample is reweighted so that the sum of weights of signal and background are the same
        """

        self.train_signal_dataset, self.test_signal_dataset, self.train_signal_weights, self.test_signal_weights = train_test_split(self.signal_dataset, self.signal_weights, test_size=0.5, random_state=42)
        self.train_background_dataset, self.test_background_dataset, self.train_background_weights, self.test_background_weights = train_test_split(self.background_dataset, self.background_weights, test_size=0.5, random_state=42)

        if reweight_background_training_sample:
            sumw_train_signal = np.sum(self.train_signal_weights)
            sumw_train_background = np.sum(self.train_background_weights)

            ratio = sumw_train_signal / sumw_train_background
            self.train_background_weights *= ratio
            self.test_background_weights *= ratio
            print("Background training sample reweighted so that sum of event weights for signal and background match. Sum of event weight = %.4f" % (np.sum(self.train_signal_weights)))

        # Create merged training and testing dataset, with targets
        self.training_dataset = np.concatenate([self.train_signal_dataset, self.train_background_dataset])
        self.training_weights = np.concatenate([self.train_signal_weights, self.train_background_weights])
        self.testing_dataset = np.concatenate([self.test_signal_dataset, self.test_background_dataset])
        self.testing_weights = np.concatenate([self.test_signal_weights, self.test_background_weights])

        # Create one-hot vector, the target of the training
        # A hot-vector is a N dimensional vector, where N is the number of classes
        # Here we assume that class 0 is signal, and class 1 is background
        # So we have [1 0] for signal and [0 1] for background
        self.training_targets = np.array([[1, 0]] * len(self.train_signal_dataset) + [[0, 1]] * len(self.train_background_dataset))
        self.testing_targets = np.array([[1, 0]] * len(self.test_signal_dataset) + [[0, 1]] * len(self.test_background_dataset))

        # Shuffle everything
        self.training_dataset, self.training_weights, self.training_targets = shuffle(self.training_dataset, self.training_weights, self.training_targets, random_state=42)
        self.testing_dataset, self.testing_weights, self.testing_targets = shuffle(self.testing_dataset, self.testing_weights, self.testing_targets, random_state=42)

    def get_training_datasets(self):
        return self.train_signal_dataset, self.train_background_dataset

    def get_testing_datasets(self):
        return self.test_signal_dataset, self.test_background_dataset

    def get_training_weights(self):
        return self.train_signal_weights, self.train_background_weights

    def get_testing_weights(self):
        return self.test_signal_weights, self.test_background_weights

    def get_training_combined_dataset_and_targets(self):
        return self.training_dataset, self.training_targets

    def get_testing_combined_dataset_and_targets(self):
        return self.testing_dataset, self.testing_targets

    def get_training_combined_weights(self):
        return self.training_weights

    def get_testing_combined_weights(self):
        return self.testing_weights

    def get_training_testing_signal_predictions(self, model, **kwargs):
        return self._get_predictions(model, self.train_signal_dataset, **kwargs), self._get_predictions(model, self.test_signal_dataset, **kwargs)

    def get_signal_predictions(self, model, **kwargs):
        return self._get_predictions(model, self.signal_dataset, **kwargs)

    def get_signal_weights(self):
        return self.signal_weights

    def get_training_testing_background_predictions(self, model, **kwargs):
        return self._get_predictions(model, self.train_background_dataset, **kwargs), self._get_predictions(model, self.test_background_dataset, **kwargs)

    def get_background_predictions(self, model, **kwargs):
        return self._get_predictions(model, self.background_dataset, **kwargs)

    def get_background_weights(self):
        return self.background_weights

    def _get_predictions(self, model, values, **kwargs):
        ignore_n_last_columns = kwargs.get('ignore_last_columns', 0)
        if ignore_n_last_columns > 0:
            values = values[:, :-ignore_n_last_columns]
        predictions = model.predict(values, batch_size=5000, verbose=1)
        return np.delete(predictions, 1, axis=1).flatten()

    def draw_inputs(self, output_dir):

        print("Plotting input variables...")
        variables = self.variables[:]

        if self.parametricDNN :
            if self.resonant_masses:
                variables += ['mass_hypothesis']
            #elif self.nonresonant_parameters_list:
            #    variables += ['k_l', 'k_t']

        for index, variable in enumerate(variables):
            output_file = os.path.join(output_dir, variable + ".pdf") 
            plotTools.drawTrainingTestingComparison(
                    training_background_data=self.train_background_dataset[:, index],
                    training_signal_data=self.train_signal_dataset[:, index],
                    testing_background_data=self.test_background_dataset[:, index],
                    testing_signal_data=self.test_signal_dataset[:, index],

                    training_background_weights=self.train_background_weights,
                    training_signal_weights=self.train_signal_weights,
                    testing_background_weights=self.test_background_weights,
                    testing_signal_weights=self.test_signal_weights,

                    x_label=variable,
                    output=output_file,
                    output_Dir=output_dir,
                    output_Name=variable
                    )

        print("Done.")

    def draw_correlations(self, output_dir):
	print("Plotting input correlation variables...")
	variables = self.variables[:]
	output_file1 = "Singal_correlation.pdf"
	output_file2 = "Background_correlation.pdf"
	
	plotTools.drawCorrelation(variables, self.signal_dataset, self.signal_weights, output_dir, output_file1)
	plotTools.drawCorrelation(variables, self.background_dataset, self.background_weights, output_dir, output_file2)
	#plotTools.drawCorrelations(self.variables[:], self.signal_dataset, output_file1, output_dir)



    #def compute_shift_values(self, parameters_list):
    #    shift = [abs(min(x)) if min(x) < 0 else 0 for x in zip(*parameters_list)]
    #    print("Shifting all non-resonant parameters by {}".format(shift))

    #    return shift

    #def _user_to_positive_parameters_list(self, parameters_list, parameters_shift=None):
    #    if not parameters_shift:
    #        self.nonresonant_parameters_shift_value = self.compute_shift_values(parameters_list)
    #    else:
    #        self.nonresonant_parameters_shift_value = parameters_shift[:]

    #    shifted_parameters_list = parameters_list[:]
    #    for i in range(len(parameters_list)):
    #        shifted_parameters_list[i] = tuple([x + self.nonresonant_parameters_shift_value[j] for j, x in enumerate(parameters_list[i])])

    #    return shifted_parameters_list

    #def _positive_to_user_parameters_list(self, parameters_list):
    #    if not self.nonresonant_parameters_shift_value:
    #        raise Exception('Cannot invert parameters transformation since _user_to_positive_parameters was not called')

    #    shifted_parameters_list = parameters_list[:]
    #    for i in range(len(parameters_list)):
    #        shifted_parameters_list[i] = tuple([x - self.nonresonant_parameters_shift_value[j] for j, x in enumerate(parameters_list[i])])

    #    return shifted_parameters_list

    #def positive_to_user_parameters(self, parameters):
    #    if not self.nonresonant_parameters_shift_value:
    #        raise Exception('Cannot invert parameters transformation since _user_to_positive_parameters was not called')

    #    return tuple([x - self.nonresonant_parameters_shift_value[j] for j, x in enumerate(parameters)])

    #def get_nonresonant_parameters_list(self):
    #    return self.nonresonant_parameters_list
    #    # return self._positive_to_user_parameters_list(self.nonresonant_parameters_list)

def get_file_from_glob(f):
    files = glob.glob(f)
    if len(files) != 1:
	print "files ",files
        raise Exception('Only one input file is supported per glob pattern: %s -> %s' % (f, files))
    return files[0]

def get_files_from_glob(f):
    files = glob.glob(f)
    if len(files) == 0:
        raise Exception('No file matching glob pattern: %s' % f)
    return files

#def draw_non_resonant_training_plots(model, dataset, output_folder, split_by_parameters=False):
#
#    # plot(model, to_file=os.path.join(output_folder, "model.pdf"))
#
#    # Draw inputs
#    output_input_plots = os.path.join(output_folder, 'inputs')
#    if not os.path.exists(output_input_plots):
#        os.makedirs(output_input_plots)
#
#    dataset.draw_inputs(output_input_plots)
#
#    training_dataset, training_targets = dataset.get_training_combined_dataset_and_targets()
#    training_weights = dataset.get_training_combined_weights()
#
#    testing_dataset, testing_targets = dataset.get_testing_combined_dataset_and_targets()
#    testing_weights = dataset.get_testing_combined_weights()
#
#    print("Evaluating model performances...")
#
#    training_signal_weights, training_background_weights = dataset.get_training_weights()
#    testing_signal_weights, testing_background_weights = dataset.get_testing_weights()
#
#    training_signal_predictions, testing_signal_predictions = dataset.get_training_testing_signal_predictions(model)
#    training_background_predictions, testing_background_predictions = dataset.get_training_testing_background_predictions(model)
#
#    print("Done.")
#
#    print("Plotting time...")
#
#    # NN output
#    plotTools.drawNNOutput(training_background_predictions, testing_background_predictions,
#                 training_signal_predictions, testing_signal_predictions,
#                 training_background_weights, testing_background_weights,
#                 training_signal_weights, testing_signal_weights,
#                 output_dir=output_folder, output_name="nn_output",form=".pdf", bins=50)
#
#    # ROC curve
#    binned_training_background_predictions, _, bins = plotTools.binDataset(training_background_predictions, training_background_weights, bins=50, range=[0, 1])
#    binned_training_signal_predictions, _, _ = plotTools.binDataset(training_signal_predictions, training_signal_weights, bins=bins)
#    plotTools.draw_roc(binned_training_signal_predictions, binned_training_background_predictions, output_dir=output_folder, output_name="roc_curve",form=".pdf")
#
#    if split_by_parameters:
#        output_folder = os.path.join(output_folder, 'splitted_by_parameters')
#        if not os.path.exists(output_folder):
#            os.makedirs(output_folder)
#
#        training_signal_dataset, training_background_dataset = dataset.get_training_datasets()
#        testing_signal_dataset, testing_background_dataset = dataset.get_testing_datasets()
#        for parameters in dataset.get_nonresonant_parameters_list():
#            user_parameters = ['{:.2f}'.format(x) for x in dataset.positive_to_user_parameters(parameters)]
#
#            print("  Plotting NN output and ROC curve for %s" % str(user_parameters))
#
#            training_signal_mask = (training_signal_dataset[:,-1] == parameters[1]) & (training_signal_dataset[:,-2] == parameters[0])
#            training_background_mask = (training_background_dataset[:,-1] == parameters[1]) & (training_background_dataset[:,-2] == parameters[0])
#            testing_signal_mask = (testing_signal_dataset[:,-1] == parameters[1]) & (testing_signal_dataset[:,-2] == parameters[0])
#            testing_background_mask = (testing_background_dataset[:,-1] == parameters[1]) & (testing_background_dataset[:,-2] == parameters[0])
#
#            p_training_background_predictions = training_background_predictions[training_background_mask]
#            p_testing_background_predictions = testing_background_predictions[testing_background_mask]
#            p_training_signal_predictions = training_signal_predictions[training_signal_mask]
#            p_testing_signal_predictions = testing_signal_predictions[testing_signal_mask]
#
#            p_training_background_weights = training_background_weights[training_background_mask]
#            p_testing_background_weights = testing_background_weights[testing_background_mask]
#            p_training_signal_weights = training_signal_weights[training_signal_mask]
#            p_testing_signal_weights = testing_signal_weights[testing_signal_mask]
#
#            suffix = format_nonresonant_parameters(user_parameters)
#            plotTools.drawNNOutput(
#                         p_training_background_predictions, p_testing_background_predictions,
#                         p_training_signal_predictions, p_testing_signal_predictions,
#                         p_training_background_weights, p_testing_background_weights,
#                         p_training_signal_weights, p_testing_signal_weights,
#                         output_dir=output_folder, output_name="nn_output_fixed_parameters_%s"%(suffix),form=".pdf", bins=50)
#
#            binned_training_background_predictions, _, bins = plotTools.binDataset(p_training_background_predictions, p_training_background_weights, bins=50, range=[0, 1])
#            binned_training_signal_predictions, _, _ = plotTools.binDataset(p_training_signal_predictions, p_training_signal_weights, bins=bins)
#            plotTools.draw_roc(binned_training_signal_predictions, binned_training_background_predictions, output_dir=output_folder, output_name="roc_curve_fixed_parameters_%s" % (suffix),form=".pdf")
#    print("Done")


####################################3
### Doing Training Here
####################################

def lr_scheduler(epoch):
    default_lr = 0.001
    drop = 0.1
    epochs_drop = 50.0
    lr = default_lr * math.pow(drop, min(1, math.floor((1 + epoch) / epochs_drop)))
    return lr

def training_resonant( variables, selection, masses , output_folder, output_model_filename):
    parametricDNN = False
    if len(masses)>1:    parametricDNN = True

# Loading Signal and Backgrounds
    dataset = DatasetManager(variables, selection, masses)

    dataset.loadResonantSignal(treename)
    dataset.loadBackgrounds(treename)
    dataset.split()

    training_dataset, training_targets = dataset.get_training_combined_dataset_and_targets()
    training_weights = dataset.get_training_combined_weights()

    testing_dataset, testing_targets = dataset.get_testing_combined_dataset_and_targets()
    testing_weights = dataset.get_testing_combined_weights()


    n_inputs = len(variables)
    if len(masses)>1: ## add mass column
        n_inputs += 1

    # You create here the real DNN structure
    model = create_resonant_model(n_inputs)

# callback: set of functions to be applied at given stages of the training procedure. You can use callbacks to get a view on internal states/statistics of model during training
    callbacks = []
    callbacks.append(ModelCheckpoint(output_model_filename, monitor='val_loss', verbose=False, save_best_only=True, mode='auto')) #Save model after every epoch
# output_logs_folder = os.path.join('hh_resonant_trained_models', 'logs', output_suffix)
# callbacks.append(keras.callbacks.TensorBoard(log_dir=output_logs_folder, histogram_freq=1, write_graph=True, write_images=False))
    callbacks.append(LearningRateScheduler(lr_scheduler)) # Provide learnign rate per epoch. lr_scheduler = have to be a function of epoch.
    #n_inputs = len(inputs)
    #if add_mass_column:
    #    n_inputs += 1

    # You create here the real DNN structure
    #model = create_resonant_model(n_inputs)

    
    start_time = timer()
    # You do the training with the compiled model
    history = model.fit(training_dataset, training_targets, sample_weight=training_weights, batch_size=batch_size, epochs=epochs,
            verbose=True, validation_data=(testing_dataset, testing_targets, testing_weights), callbacks=callbacks)

    end_time = timer()
    training_time = datetime.timedelta(seconds=(end_time - start_time))

    save_training_parameters(output_folder, model,
            batch_size=batch_size, epochs=epochs,
            training_time=str(training_time),
            masses=masses,
            with_mass_column=parametricDNN,
            inputs=variables,
            cut=selection)

    plotTools.draw_keras_history(history, output_dir=output_folder, output_name="loss.pdf")

    # Save history
    print("Saving model training history...")
    output_history_filename = 'hh_resonant_trained_model_history.pklz'
    output_history_filename = os.path.join(output_folder, output_history_filename)
    with gzip.open(output_history_filename, 'wb') as f:
        pickle.dump(history.history, f)
    print("Done.")

    print("All done. Training time: %s" % str(training_time))

    return dataset, model

     

####################################3
### Ploting for Training Here
####################################

def draw_resonant_training_plots(model, dataset, output_folder, split_by_mass=False):
    #step0 Draw inputs
    output_input_plots = os.path.join(output_folder, 'inputs')
    if not os.path.exists(output_input_plots):
        os.makedirs(output_input_plots)
    
    dataset.draw_inputs(output_input_plots)
    dataset.draw_correlations(output_folder)

    #step1 create training and testing dataset/weight
    training_dataset, training_targets = dataset.get_training_combined_dataset_and_targets()
    training_weights = dataset.get_training_combined_weights()

    testing_dataset, testing_targets = dataset.get_testing_combined_dataset_and_targets()
    testing_weights = dataset.get_testing_combined_weights()

    print("Evaluating model performances...")

    training_signal_weights, training_background_weights = dataset.get_training_weights()
    testing_signal_weights, testing_background_weights = dataset.get_testing_weights()

    #step2 evaludation, get the predictions
    training_signal_predictions, testing_signal_predictions = dataset.get_training_testing_signal_predictions(model)
    training_background_predictions, testing_background_predictions = dataset.get_training_testing_background_predictions(model)

    print("Done.")

    print("Plotting time...")

    # NN output
    plotTools.drawNNOutput(training_background_predictions, testing_background_predictions,
                 training_signal_predictions, testing_signal_predictions,
                 training_background_weights, testing_background_weights,
                 training_signal_weights, testing_signal_weights,
                 output_dir=output_folder, output_name="nn_output",form=".pdf", bins=50)

    # ROC curve
    binned_training_background_predictions, _, bins = plotTools.binDataset(training_background_predictions, training_background_weights, bins=50, range=[0, 1])
    binned_training_signal_predictions, _, _ = plotTools.binDataset(training_signal_predictions, training_signal_weights, bins=bins)
    plotTools.draw_roc(binned_training_signal_predictions, binned_training_background_predictions, output_dir=output_folder, output_name="roc_curve",form=".pdf")

    #if split_by_mass:
    if dataset.parametricDNN and split_by_mass:
        output_folder = os.path.join(output_folder, 'splitted_by_mass')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        training_signal_dataset, training_background_dataset = dataset.get_training_datasets()
        testing_signal_dataset, testing_background_dataset = dataset.get_testing_datasets()
        for m in dataset.resonant_masses:
            print("  Plotting NN output and ROC curve for M=%d" % m)

            training_signal_mask = training_signal_dataset[:,-1] == m
            training_background_mask = training_background_dataset[:,-1] == m
            testing_signal_mask = testing_signal_dataset[:,-1] == m
            testing_background_mask = testing_background_dataset[:,-1] == m

            p_training_background_predictions = training_background_predictions[training_background_mask]
            p_testing_background_predictions = testing_background_predictions[testing_background_mask]
            p_training_signal_predictions = training_signal_predictions[training_signal_mask]
            p_testing_signal_predictions = testing_signal_predictions[testing_signal_mask]

            p_training_background_weights = training_background_weights[training_background_mask]
            p_testing_background_weights = testing_background_weights[testing_background_mask]
            p_training_signal_weights = training_signal_weights[training_signal_mask]
            p_testing_signal_weights = testing_signal_weights[testing_signal_mask]
            plotTools.drawNNOutput(
                         p_training_background_predictions, p_testing_background_predictions,
                         p_training_signal_predictions, p_testing_signal_predictions,
                         p_training_background_weights, p_testing_background_weights,
                         p_training_signal_weights, p_testing_signal_weights,
                         output_dir=output_folder, output_name="nn_output_fixed_M%d"% (m), form=".pdf",
                         bins=50)

            binned_training_background_predictions, _, bins = plotTools.binDataset(p_training_background_predictions, p_training_background_weights, bins=50, range=[0, 1])
            binned_training_signal_predictions, _, _ = plotTools.binDataset(p_training_signal_predictions, p_training_signal_weights, bins=bins)
            plotTools.draw_roc(binned_training_signal_predictions, binned_training_background_predictions, output_dir=output_folder, output_name="roc_curve_fixed_M_%d" % (m),form=".pdf")

    print("Done")

def save_training_parameters(output, model, **kwargs):
    parameters = {
            'extra': kwargs
            }

    model_definition = model.to_json()
    m = json.loads(model_definition)
    parameters['model'] = m

    with open(os.path.join(output, 'parameters.json'), 'w') as f:
        json.dump(parameters, f, indent=4)


def export_for_lwtnn(model, name):
    base, _ = os.path.splitext(name)

    # Export architecture of the model
    with open(base + '_arch.json', 'w') as f:
        f.write(model.to_json())

    # And the weights
    model.save_weights(base + "_weights.h5")


if __name__ == "__main__":
#sfile = "/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/radion_M400_all.root"
#bfile = "/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/TT_all.root"

### goal, Parametric and Dedicated DNN : 
### kinematics + MT+MT2
### kinematics + MT+MT2+MJJ
### kinematics + MT+MT2+HME
### kinematics + MT+MT2+MJJ+HME

#features_store   = ['jj_pt','ll_pt','ll_M','ll_DR_l_l','jj_DR_j_j','llmetjj_DPhi_ll_jj','llmetjj_minDR_l_j','llmetjj_MTformula','hme_h2mass_reco','isSF', "mt2"]
    features_store   = ['jj_pt','ll_pt','ll_M','ll_DR_l_l','jj_DR_j_j','llmetjj_DPhi_ll_jj','llmetjj_minDR_l_j','llmetjj_MTformula','isSF', "mt2"] ##noHME
    features_store   = ['jj_pt','ll_pt','ll_M','ll_DR_l_l','jj_DR_j_j','llmetjj_DPhi_ll_jj','llmetjj_minDR_l_j','llmetjj_MTformula','isSF', 'mt2','hme_h2mass_reco','jj_M'] ##with HME
    features_store   = ['isSF', 'jj_pt','ll_pt','ll_M','ll_DR_l_l','jj_DR_j_j','llmetjj_DPhi_ll_jj','llmetjj_minDR_l_j','llmetjj_MTformula', 'mt2','hme_h2mass_reco','jj_M'] ##with HME
    cut = "91-ll_M>15 && hme_h2mass_reco>=250"
    mass_list = [400]
    mass_list = [400, 450, 500, 550, 600, 650]
    parametricDNN = (len(mass_list) != 1)

    variablename = "MTandMT2MjjHME"
    suffix = str(epochs)+"epochs"
    output_suffix = '{}_{:%Y-%m-%d}_{}'.format(variablename, datetime.date.today(), suffix)
    if parametricDNN:
        output_suffix = output_suffix+"_paramtricDNN"
    else:
        output_suffix = output_suffix+"_dedicatedDNN%d"%mass_list[0]

    output_folder = os.path.join('DNNmodels', output_suffix)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_model_filename = 'hh_resonant_trained_model.h5'
    output_model_filename = os.path.join(output_folder, output_model_filename)

    n_inputs = len(features_store)
    dataset, model = training_resonant(features_store, cut, mass_list , output_folder, output_model_filename)


    export_for_lwtnn(model, output_model_filename)
# Draw the inputs 
    draw_resonant_training_plots(model, dataset, output_folder, split_by_mass=parametricDNN)
    ###
    #model = keras.models.load_model(output_model_filename)

