import os
import pandas as pd
import numpy as np
from   sklearn.model_selection import train_test_split
from   sklearn.utils import shuffle
from   keras.utils import to_categorical
import configuration as conf
'''
Goal: saving csv files for test, training, and validation samples. For signal and background, and for each mass.
'''
# Parameters
data_folder = conf.data_folder
all_masses  = [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 800, 900]
features    = conf.features_store
weights     = conf.weights

for mass in all_masses:
    print '----- MASS', mass, '-----'
    df_s = pd.read_csv(data_folder + '/signal' + str(mass) + '_scaled.csv')
    df_b = pd.read_csv(data_folder + '/background' + str(mass) + '_scaled.csv')
    # Those functions simple returns the trained/tested dataset/target
    train_signal_dataset,      test_signal_dataset,     train_signal_weights,     test_signal_weights     = train_test_split(df_s[features], df_s[weights], test_size=0.2, random_state=42)
    train_background_dataset,  test_background_dataset, train_background_weights, test_background_weights = train_test_split(df_b[features], df_b[weights], test_size=0.2, random_state=42)
    sumw_train_signal         = np.sum(train_signal_weights)
    sumw_train_background     = np.sum(train_background_weights)
    ratio                     = sumw_train_signal / sumw_train_background
    train_background_weights *= ratio
    test_background_weights  *= ratio
    print "Background training sample reweighted so that sum of event weights for signal and background match:"
    print " Total sum of Signal events weight =",     np.sum(train_signal_weights)
    print " Total sum of Background events weight =", np.sum(train_background_weights)
    # Create merged training and testing dataset, with targets
    training_dataset = np.concatenate([train_signal_dataset, train_background_dataset])
    training_weights = np.concatenate([train_signal_weights, train_background_weights])
    testing_dataset  = np.concatenate([test_signal_dataset,  test_background_dataset])
    testing_weights  = np.concatenate([test_signal_weights,  test_background_weights])
    # Create the target of the training. Basically [0] for signal and [1] for background
    training_targets = np.array([[0]] * len(train_signal_dataset) + [[1]] * len(train_background_dataset))
    testing_targets  = np.array([[0]] * len(test_signal_dataset) + [[1]] * len(test_background_dataset))
    training_targets = to_categorical(training_targets) #[1,0]=Signal ; [0,1]=Backgrouns
    testing_targets  = to_categorical(testing_targets) #[1,0]=Signal ; [0,1]=Backgrouns
    
    # Shuffle them
    training_dataset, training_weights, training_targets = shuffle(training_dataset, training_weights, training_targets, random_state=42)
    testing_dataset, testing_weights, testing_targets    = shuffle(testing_dataset, testing_weights, testing_targets, random_state=42)
 
    # Now create the validation smapels. Validation is used for optimizing hyperparmaters, but final score is measured on test sample.
    trainFinal_dataset, valid_dataset, trainFinal_weights, valid_weights, trainFinal_targets, valid_targets = train_test_split(training_dataset, training_weights, training_targets, test_size=0.2, random_state=40)
 
    #Save them into a csv file
    print 'Saving Training/Testing datasets'
    pd.DataFrame(data=trainFinal_dataset, columns=features).to_csv(data_folder + '/train_dataset' + str(mass) + '.csv')
    pd.DataFrame(data=trainFinal_weights, columns=weights).to_csv(data_folder + '/train_weights' + str(mass) + '.csv')
    pd.DataFrame(data=trainFinal_targets, columns=['is_s','is_b']).to_csv(data_folder + '/train_targets' + str(mass) + '.csv')
    pd.DataFrame(data=valid_dataset, columns=features).to_csv(data_folder + '/valid_dataset' + str(mass) + '.csv')
    pd.DataFrame(data=valid_weights, columns=weights).to_csv(data_folder + '/valid_weights' + str(mass) + '.csv')
    pd.DataFrame(data=valid_targets, columns=['is_s','is_b']).to_csv(data_folder + '/valid_targets' + str(mass) + '.csv')
    pd.DataFrame(data=testing_dataset,  columns=features).to_csv(data_folder + '/test_dataset' + str(mass) + '.csv')
    pd.DataFrame(data=testing_weights,  columns=weights).to_csv(data_folder + '/test_weights' + str(mass) + '.csv')
    pd.DataFrame(data=testing_targets,  columns=['is_s','is_b']).to_csv(data_folder + '/test_targets' + str(mass) + '.csv')
print "Done!"
