import os
import sys
import pandas as pd
import numpy as np
import json
import colorsys
from   sklearn.metrics import auc
import matplotlib.pyplot as plt
import common as com
import configuration as conf
'''
Goal: compare different models.
'''
# Parameters
folder_name = conf.comp_folder_name
folders     = []
for iTmp in range(len(sys.argv)-1):
    folders.append(sys.argv[iTmp+1])
Ncolor      = len(folders) 
HSV_tuples  = [(x*1.0/Ncolor, 0.5, 0.5) for x in range(Ncolor)]
RGB_tuples  = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

SortedFolders = []
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
for i, folder in enumerate(folders):
    tested    = folder.split("_")
    tested_id = str('L'+tested[4][-1] + '_' + 'N'+tested[5][-2:] + '_' + tested[6] + '_' + tested[7])
    tpr       = np.load(folder + '/numpy/tpr.npy' )
    fpr       = np.load(folder + '/numpy/fpr.npy' )   
    auc_keras = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=tested_id+' (A={:.3f})'.format(auc_keras), color=RGB_tuples[i])
    auc_rounded = '%.4f' % auc_keras
    SortedFolders.append((auc_rounded, tested_id))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig(folder_name+'/ROC_all.png')
plt.clf()
print 'Sorted Folders:'
for item in sorted(SortedFolders, key=lambda x: x[0]):
    print item
arr_epochs, arr_name = [], []
for i, folder in enumerate(folders):
    with open(folder + '/model_history.json') as f:
        history = json.load(f)
    tested    = folder.split("_")
    arr_epochs.append(len(history['loss']))
    arr_name.append( str('L'+tested[4][-1] + '_' + 'N'+tested[5][-2:] + '_' + tested[6]) + '_' + tested[7] ) 
plt.hist(arr_epochs, label='Epochs', color='blue')
plt.xlabel('Number of Epochs')
plt.savefig(folder_name+'/Epochs.png')
plt.clf()
print 'DONE!'
