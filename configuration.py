import datetime
import tensorflow as tf
now = datetime.datetime.now()

# Basic parameters
input_files      = '/Users/Luca2/Downloads/ML/HHNTuples/HHNtuple_20170814_10k/'
data_folder      = 'data/CSV/'
masses           = [450]
selection        = "(91 - ll_M) > 15"
resonant_weights = {'__base__': "event_weight * trigeff * jjbtag_heavy * jjbtag_light * llidiso * pu",
                    'DY.*'    : "dy_nobtag_to_btagM_weight"}
features         = ['jj_pt','ll_pt','ll_M','ll_DR_l_l','jj_DR_j_j','llmetjj_DPhi_ll_jj','llmetjj_minDR_l_j','llmetjj_MTformula']
features_toBeSca = ['jj_pt','ll_pt','ll_M','ll_DR_l_l','jj_DR_j_j','llmetjj_DPhi_ll_jj','llmetjj_minDR_l_j','llmetjj_MTformula','hme_h2mass_reco','hme_entries_reco']
features_store   = ['jj_pt','ll_pt','ll_M','ll_DR_l_l','jj_DR_j_j','llmetjj_DPhi_ll_jj','llmetjj_minDR_l_j','llmetjj_MTformula','hme_h2mass_reco','hme_entries_reco','isSF']
range_feature    = {'jj_pt':[50,0,400], 'll_pt':[50,0,400], 'll_M':[20,0,80], 'll_DR_l_l':[5,0,5], 'jj_DR_j_j':[5,0,5], 'llmetjj_DPhi_ll_jj':[6,0,3], 'llmetjj_minDR_l_j':[5,0,5], 'llmetjj_MTformula':[25,0,400], 'hme_h2mass_reco':[50,100.,1000], 'hme_entries_reco':[30,0.,10], 'isSF':[2,-0.5,1.5]}
features_PCA     = ['PC_0','PC_1','PC_2','PC_3','PC_4','PC_5','PC_6','PC_7']
features_LDA     = ['LDA_0','LDA_1','LDA_2','LDA_3','LDA_4','LDA_5','LDA_6','LDA_7']
weights          = ['weight']

# Training parameters
base_folder      = "Hhh_"
output_suffix    = "epoch_M" #str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_Time_' + str(now.hour) + '_' + str(now.minute) + '_' + str(now.second) # if train = False it will analze models in thos folder
model_name       = "model.ckpt"
training         = True
usePCA           = True
useLDA           = True

# Model paramters
lr               = 0.0005
regular          = tf.keras.regularizers.l2()
training_epochs  = 50
batch_size       = 1500

# Testing parameters
numpy_folder     = 'numpy'
signal_color     = '#468966'
background_color = '#B64926'

# Comparison
comp_folder_name = 'Comparison'
