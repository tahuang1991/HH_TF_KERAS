#!/bin/bash
#Run DNN changin all hyperparameters
#mass=400
#declare -a layers=(2) #3)
#declare -a nodes=(60) # 100)
#declare -a lrs=(0.0002 0.0005 0.0007)
#declare -a beta1=(0.9 0.85)
#declare -a beta2=(0.999)
#declare -a epsilon=(1e-08)
#declare -a decay=(0 1)
#declare -a optimiz=('Adam' 'GDS')
#declare -a learning
#declare -a regul=('l25', l15...)
#declare -a batch_sizes=(1500)
#
#for la in "${layers[@]}"
#do
#    for no in "${nodes[@]}"
#    do
#        for lr in "${lrs1[@]}"
#        do
#            for batch in "${batch_sizes[@]}"
#            do
#                python 3_TrainModel_opt.py $mass $la $no $lr $batch
#            done
#        done
#    done
#done
#
# Compute performances for each of them
declare -a folders=(
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0004_batch100
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0004_batch300
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0004_batch600
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0004_batch1000
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0004_batch1500
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0007_batch100
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0007_batch300
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0007_batch600
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0007_batch1000
Hhh_50epoch_M400_opt_layers2_nodes60_lr0.0007_batch1500)

#for fol in "${folders[@]}"
#do
#    python 4_TestModel.py 400 $fol
#done

#Compare performances
args=''
for fol in "${folders[@]}"
do
    if [[ $fol != *"0az07"* ]]; then
        args="$args $fol"
    fi
done
echo $args
python 5_Comparison.py $args
