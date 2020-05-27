import ROOT

# Select Theano as backend for Keras
from os import environ
environ['KERAS_BACKEND'] = 'theano'

# Set architecture of system (AVX instruction set is not supported on SWAN)
#environ['THEANO_FLAGS'] = 'gcc.cxxflags=-march=corei7'

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam


import configuration as conf

sfile = "/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/radion_M400_all.root"
bfile = "/Users/taohuang/Documents/DiHiggs/20180205_20180202_10k_Louvain_ALLNoSys/TT_all.root"

treename = "t"

### get signal and background ttree
stfile = ROOT.TFile.Open(sfile); stree = stfile.Get(treename)
btfile = ROOT.TFile.Open(bfile); btree = btfile.Get(treename)

#features_store   = ['jj_pt','ll_pt','ll_M','ll_DR_l_l','jj_DR_j_j','llmetjj_DPhi_ll_jj','llmetjj_minDR_l_j','llmetjj_MTformula','hme_h2mass_reco','isSF', "mt2"]
features_store   = ['jj_pt','ll_pt','ll_M','ll_DR_l_l','jj_DR_j_j','llmetjj_DPhi_ll_jj','llmetjj_minDR_l_j','llmetjj_MTformula','isSF', "mt2"] ##noHME

dataloader = ROOT.TMVA.DataLoader('dataset_pymva')


numVariables = len(features_store)
for var in features_store:
    dataloader.AddVariable(var)

dataloader.AddSignalTree(stree, 1.0)
dataloader.AddBackgroundTree(btree, 1.0)


##set event weight
dataloader.SetSignalWeightExpression("total_weight")
dataloader.SetBackgroundWeightExpression("total_weight")


trainTestSplit = 0.5


dataloader.PrepareTrainingAndTestTree(ROOT.TCut('(91 - ll_M) > 15'),
        'TrainTestSplit_Signal={}:'.format(trainTestSplit)+\
        'TrainTestSplit_Background={}:'.format(trainTestSplit)+\
        'SplitMode=Random')

ROOT.TMVA.Tools.Instance()
ROOT.TMVA.PyMethodBase.PyInitialize()

outputFile = ROOT.TFile.Open('TMVAOutputPyMVA.root', 'RECREATE')

factory = ROOT.TMVA.Factory('TMVAClassification', outputFile,
        '!V:!Silent:Color:DrawProgressBar:Transformations=I,G:'+\
        'AnalysisType=Classification')


# Create model
model = Sequential()
#model = tf.keras.Sequential()

#model.add(Dense(60, kernel_initializer="glorot_uniform", activation="relu", input_dim=numVariables, kernel_regularizer=conf.regular))
#model.add(Dense(60, kernel_initializer="glorot_uniform", activation="relu"))
#model.add(Dense(2, kernel_initializer="glorot_uniform", activation="softmax"))


# kernel_initializer: Initializations define the way to set the initial random weights of Keras layers (glorot_uniform = uniform initializer)
# activation: activation function for the nodes. relu is good for intermedied step, softmax for the last one.
model.add(Dense(100, kernel_initializer="glorot_uniform", activation="relu", input_dim=numVariables))
n_hidden_layers = 4
for i in range(n_hidden_layers):
    model.add(Dense(100, kernel_initializer="glorot_uniform", activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2, kernel_initializer="glorot_uniform", activation="softmax"))
# You compile it so you can actually use it

# Compile the model (before training a model, you need to configure the learning process)
#model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=conf.lr), metrics=[com.sensitivity, com.specificity])
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=conf.lr), metrics=['categorical_accuracy',])

# Print summary of model
model.summary()
#modelfile = "models/hh_resonant_trained_models_kinematicwithMTandMT.h5"
model.save('model.h5')


# Keras interface with previously defined model
factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras, 'PyKeras',
        'H:!V:VarTransform=G:FilenameModel=model.h5:'+\
        'NumEpochs=10:BatchSize=32:'+\
        'TriesEarlyStopping=3')


# Gradient tree boosting from scikit-learn package
factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyGTB, 'GTB',
        'H:!V:VarTransform=None:'+\
        'NEstimators=100:LearningRate=0.1:MaxDepth=3')

factory.TrainAllMethods()

factory.TestAllMethods()

roc_canvas = factory.GetROCCurve(dataloader)
roc_canvas.Draw()

