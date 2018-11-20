import sys
import os
import pandas as pd
from   sklearn.preprocessing import StandardScaler
from   sklearn.decomposition import PCA
#from   sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from   sklearn.lda import LDA
import matplotlib.pyplot as plt
import configuration as conf
'''
Goal: perform PCA analysis.
'''
# Parameters
data_folder     = conf.data_folder
mass            = sys.argv[1]
features        = conf.features
n_inputs        = len(features)
print '---- MASS', mass, '----'
df_train        = pd.read_csv(data_folder + '/train_dataset' + str(mass) + '.csv')
df_train_target = pd.read_csv(data_folder + '/train_targets' + str(mass) + '.csv')
df_valid        = pd.read_csv(data_folder + '/valid_dataset' + str(mass) + '.csv')
df_test         = pd.read_csv(data_folder + '/test_dataset' + str(mass) + '.csv')
folder_name     = conf.base_folder + str(conf.training_epochs) + conf.output_suffix + str(mass) + '/'
if not os.path.exists(folder_name + '/PCA'):
    os.makedirs(folder_name + '/PCA')

x       = df_train[features].values
x_valid = df_valid[features].values
x_test  = df_test[features].values
# Let's get a feeling on how the 2 classes are distributes along the different features: let us visualize them via histograms.
pca = PCA(n_components=2) # You should Standardize the features already, but I did it already
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df_train_target['is_b']], axis = 1) # 0:HH, 1:TT

#Visualize 2D projection
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1, 0]
colors  = ['r', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['is_b'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c = color, alpha=0.5,
               s = 50)
ax.legend(['TT', 'HH'])
ax.grid()
plt.savefig(folder_name + '/PCA/PCA_2Components.png')

# Now let's Apply PCA but let's keep all features
cols = []
for tmp_num in range(n_inputs):
    cols.append('PC_'+str(tmp_num))
# PCA on all train/valid/test samples
pca_tot = PCA(n_components=len(features)) # You should Standardize the features already, but I did it already
pca_train = pca_tot.fit_transform(x)
pcaDf_train = pd.DataFrame(data = pca_train, columns = cols)
pca_valid = pca_tot.transform(x_valid)
pcaDf_valid = pd.DataFrame(data = pca_valid, columns = cols)
pca_test = pca_tot.transform(x_test)
pcaDf_test = pd.DataFrame(data = pca_test, columns = cols)
print 'The transformed features have variance:', pca_tot.explained_variance_ratio_
# Save DF after PCA
pd.DataFrame(data=pcaDf_train, columns=cols).to_csv(data_folder + '/train_dataset' + str(mass) + '_PCA.csv')
pd.DataFrame(data=pcaDf_valid, columns=cols).to_csv(data_folder + '/valid_dataset' + str(mass) + '_PCA.csv')
pd.DataFrame(data=pcaDf_test, columns=cols).to_csv(data_folder + '/test_dataset' + str(mass) + '_PCA.csv')
