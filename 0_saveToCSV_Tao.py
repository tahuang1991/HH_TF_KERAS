import os
import re
import numpy as np
import pandas as pd
import common as com
import configuration as conf
from localSamplelist import * 
'''
Goal: converting each S and B root file into a csv file.
'''
# Parameters
#input_files = conf.input_files
data_folder = conf.data_folder      
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
all_masses    = [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 750, 800, 900]
features      = conf.features_store
print "features ",features
#resonant_weights = conf.resonant_weights
resonant_weights = "final_total_weight"
for shortname in full_local_samplelist.keys():    
    for samplename in full_local_samplelist[shortname].keys():
        f = full_local_samplelist[shortname][samplename]['path']
        print "file ",f
        treename = "evtreeHME_nn"
        if "Radion" in samplename: treename = "Friends"
        #cross_section = full_local_samplelist[shortname][samplename]['cross_section']
        #event_weight_sum = full_local_samplelist[shortname][samplename]['event_weight_sum']
        dataset_b, weight_b = com.tree_to_numpy(f, treename, features, resonant_weights,  cut=conf.selection, reweight_to_cross_section=True)
        print "dataset_b ",dataset_b," type ",type(dataset_b)
        df_b = pd.DataFrame(data=dataset_b, columns=features)
        #print dataset_b["jj_pt"].shape
        #df_b = pd.DataFrame(data = dataset_b["jj_pt"], columns = ["jj_pt"])
        df_b['weight'] = weight_b
        df_b.to_csv(data_folder + '/%s_%s.csv'%(shortname, samplename))
        break
