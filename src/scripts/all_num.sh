#!/bin/bash


while getopts d:g:m flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        g) device=${OPTARG};;
        m) mcaflag=1;;
    esac
done

[ -z "$dataset" ] && dataset="epinions"
[ -z "$device" ] && device=0
[ -z "$mcaflag" ] && mcaflag=0

TASK=('none' 'pga' 'srwa' 'rev' 'trial') 
IA_METHOD=('random' 'popular' 'bops')
MODEL=('consis')
NUM=(3 4 5)

if [ "$mcaflag" -eq "1" ]
then
    echo "run mca num, press anything to confirm"
    read
    for model in "${MODEL[@]}"
    do
        for num in "${NUM[@]}"
        do 
            python main.py --nop $num --model $model --dataset $dataset --task mca --method msops --blist 5 --device $device
        done
    done
    exit 0
fi

echo "run num except mca"

for model in "${MODEL[@]}"
do
    for num in "${NUM[@]}"
    do 
        python main.py --nop $num --model $model --dataset $dataset --task ca --method bops --blist 5 --device $device

        for task in "${TASK[@]}"
        do 
            python main.py --nop $num --model $model --dataset $dataset --task $task --blist 5 --device $device
        done
        for method in "${IA_METHOD[@]}"
        do 
            python main.py --nop $num --model $model --dataset $dataset --task ia --method $method --blist 5 --device $device
        done 
    done
done


