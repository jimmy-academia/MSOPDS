#!/bin/bash

# bash all.sh -d epinions -g 0 -a

while getopts d:g:am flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        g) device=${OPTARG};;
        a) run_one=1;;
        m) mcaflag=1;;
    esac
done

[ -z "$dataset" ] && dataset="epinions"
[ -z "$device" ] && device=0
[ -z "$run_one" ] && run_one=0
[ -z "$mcaflag" ] && mcaflag=0

TASK=('none' 'pga' 'srwa' 'rev' 'trial') 
IA_METHOD=('random' 'popular' 'bops')
MODEL=('consis')

# # nop = 1
if [ "$run_one" -eq "1" ]
then
    echo "(re)run 1p, press anything to confirm"
    read

    python main.py --nop 1 --model consis --dataset $dataset --task ca --method bops --device $device
    for method in "${IA_METHOD[@]}"
    do 
        python main.py --nop 1 --model consis --dataset $dataset --task ia --method $method --device $device
    done 
fi

if [ "$mcaflag" -eq "1" ]
then
    echo "run mca 2p, press anything to confirm"
    read
    for model in "${MODEL[@]}"
    do
        python main.py --nop 2 --model $model --dataset $dataset --task mca --method msops --device $device
    done
    exit 0
fi

echo "run 2p except mca"
# nop = 2
for model in "${MODEL[@]}"
do
    for task in "${TASK[@]}"
    do 
        python main.py --nop 2 --model $model --dataset $dataset --task $task --device $device
    done
    for method in "${IA_METHOD[@]}"
    do 
        python main.py --nop 2 --model $model --dataset $dataset --task ia --method $method --device $device
    done 
    python main.py --nop 2 --model $model --dataset $dataset --task ca --method bops --device $device
done


