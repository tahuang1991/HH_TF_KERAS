#!/bin/bash
declare -a arr=(260 270 300 350 400 450 500 550 600 650 800 900)
#python 0_saveToCSV.py
#python 1_PlotSandB.py
#python 2_CreateTrainTest.py

#for i in "${arr[@]}"
#do
#   python 3_TrainModel.py "$i"
#done

for i in "${arr[@]}"
do
   python 4_TestModel.py "$i"
done

python 5_Comparison.py folder_name1 folder_name2 folder_name3
