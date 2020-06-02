# Instructions
## Train on a single mass value
. python 0_saveToCSV.py
. python 1_PlotSandB.py
. python 2_CreateTrainTest.py
. python 3_TrainModel.py mass_value
. pyhton 3b_PCA.py mass_value # optional
. python 4_TestModel.py folder_name

## Compare DNN performances
. python 5_Comparison.py folder_name1 folder_name2

## Train all masses individually
. source SendAll.sh

## Optimization of hyperparamters
. source Optimization.sh
> We learned that: layers=2,3, nodes=60,100, lr<=0.0004
> We prefer batch=1500
> softmax better than sigmoid that gives non peaking S DNN output

# TODO LIST
. Use autoencoding
. Jumping frog, i.e. from model get weights and biases, compute distsane from 0,0,0, and then reinitialize at the same distance from 0,0,0.


##New framework since 2020

. python kerasRDataFrame.py to do training and ploting 
. python addNNToTree.py to add prediction to the tree

DNNModelLUT.py is a dictionary for DNN models and inputs
