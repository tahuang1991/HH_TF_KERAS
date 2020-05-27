import os
import re
import numpy as np
import pandas as pd
import common as com
import configuration as conf
'''
Goal: converting each S and B root file into a csv file.
'''
# Parameters
input_files = conf.input_files
data_folder = conf.data_folder      
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
all_masses    = [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 800, 900]
features      = conf.features_store
#resonant_weights = conf.resonant_weights
resonant_weights = "final_total_weight"

# Loading Signal and Backgrounds
for mass in all_masses:
    print '----- MASS', mass, '-----'
    f = input_files + '/radion_M' + str(mass) + '.root'
    dataset, weight = com.tree_to_numpy(f, features, conf.resonant_weights, cut=conf.selection, reweight_to_cross_section=True)
    df_s = pd.DataFrame(data=dataset, columns=features)
    df_s['weight'] = weight
    df_s.to_csv(data_folder + '/signal' + str(mass) + '.csv')
    
    f = input_files + '/TT.root'
    dataset_b, weight_b = com.tree_to_numpy(f, features, conf.resonant_weights, cut=conf.selection, reweight_to_cross_section=True)
    df_b = pd.DataFrame(data=dataset_b, columns=features)
    df_b['weight'] = weight_b
    df_b.to_csv(data_folder + '/background.csv')
