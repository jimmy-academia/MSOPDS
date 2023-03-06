#!/bin/bash

# bash scripts/all_opb.sh -d epinions -g 0

while getopts d:g:b:m flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        g) device=${OPTARG};;
        m) mcaflag=1;;
        b) bnum=${OPTARG};;
    esac
done

[ -z "$dataset" ] && dataset="epinions"
[ -z "$device" ] && device=0
[ -z "$mcaflag" ] && mcaflag=0
[ -z "$bnum" ] && bnum=5

echo "opb with b = $bnum"

TASK=('none' 'pga' 'srwa' 'rev' 'trial') 
IA_METHOD=('random' 'popular' 'bops')
MODEL=('consis')
OPB=(3 4 5)


if [ "$mcaflag" -eq "1" ]
then
    echo "run mca opb, press anything to confirm"
    read
    for model in "${MODEL[@]}"
    do
        for opb in "${OPB[@]}"
        do 
            python main.py --nop 2 --model $model --dataset $dataset --task mca --method msops --blist $bnum --opb $opb --device $device
        done
    done
    exit 0
fi

echo "run opb except mca"

for model in "${MODEL[@]}"
do
    for opb in "${OPB[@]}"
    do 
        python main.py --nop 2 --model $model --dataset $dataset --task ca --method bops --blist $bnum --opb $opb --device $device

        for task in "${TASK[@]}"
        do 
            python main.py --nop 2 --model $model --dataset $dataset --task $task --blist $bnum --opb $opb --device $device
        done
        for method in "${IA_METHOD[@]}"
        do 
            python main.py --nop 2 --model $model --dataset $dataset --task ia --method $method --blist $bnum --opb $opb --device $device
        done 
    done
done


