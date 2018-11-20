import os
import pandas as pd
import matplotlib.pyplot as plt
from   sklearn.preprocessing import StandardScaler
import configuration as conf
'''
Goal: plotting physics variable for each S and B csv file (original, and scaled).
'''
# Parameters
data_folder = conf.data_folder      
all_masses  = [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 800, 900]
range_feature = conf.range_feature
for mass in all_masses:
    print '----- MASS', mass, '-----'
    df_s = pd.read_csv(data_folder + '/signal' + str(mass) + '.csv')
    df_b = pd.read_csv(data_folder + '/background.csv')
    if not os.path.exists(data_folder + '/PNG_' + str(mass) + '/'):
        os.makedirs(data_folder + '/PNG_' + str(mass) + '/')
    # 1D Plots
    print '1D plotting'
    for col in conf.features_store:
        bins  = range_feature[col][0]
        mymin = range_feature[col][1]
        mymax = range_feature[col][2]
        plt.hist(df_s[col], bins, range=(mymin,mymax), label='S', fill=False, edgecolor='red', color='red', normed=True, alpha=1)
        plt.hist(df_b[col], bins, range=(mymin,mymax), label='B', fill=False, edgecolor='blue', color='blue', normed=True, alpha=1)
        plt.xlabel(col)
        plt.legend(loc='best')
        plt.savefig(data_folder + '/PNG_' + str(mass) + '/' + col + '.png')
        plt.clf()
    # Correlation
    plt.matshow(df_s[conf.features].corr(), vmin=-1, vmax=1, cmap='seismic') # It plots a matrix
    plt.xticks(range(len(conf.features)), conf.features)
    plt.yticks(range(len(conf.features)), conf.features)
    plt.colorbar()
    plt.xticks(rotation=90)
    plt.savefig(data_folder + '/PNG_' + str(mass) + '/corr_s.png', bbox_inches = "tight")
    plt.clf()
    plt.matshow(df_b[conf.features].corr(), vmin=-1, vmax=1, cmap='seismic')
    plt.xticks(range(len(conf.features)), conf.features)
    plt.yticks(range(len(conf.features)), conf.features)
    plt.xticks(rotation=90)
    plt.colorbar()
    plt.savefig(data_folder + '/PNG_' + str(mass) + '/corr_b.png', bbox_inches = "tight")
    plt.clf()
    # Scaler
    print 'Scaling dataframes'
    scaler = StandardScaler()
    df_s[conf.features_toBeSca] = scaler.fit_transform(df_s[conf.features_toBeSca]) # Fit so you remember how to scale
    df_b[conf.features_toBeSca] = scaler.transform(df_b[conf.features_toBeSca])     # Apply the same scaling to the background
    df_s.to_csv(data_folder + '/signal' + str(mass) + '_scaled.csv')
    df_b.to_csv(data_folder + '/background' + str(mass) + '_scaled.csv')
    # 1D Plots
    print '1D plotting'
    for col in conf.features_toBeSca:
        bins  = range_feature[col][0]
        plt.hist(df_s[col], bins, label='S', fill=False, edgecolor='red', color='red', normed=True, alpha=1)
        plt.hist(df_b[col], bins, label='B', fill=False, edgecolor='blue', color='blue', normed=True, alpha=1)
        plt.xlabel(col)
        plt.legend(loc='best')
        plt.savefig(data_folder + '/PNG_' + str(mass) + '/scaled_' + col + '.png')
        plt.clf()
print 'DONE!'
# You could add more masses on the same plot
