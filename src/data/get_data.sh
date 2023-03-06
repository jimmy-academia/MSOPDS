#!/bin/bash

if [ -d "cache" ] 
then
    echo "'cache' directory already exists, press any key to overwrite";
    read -t 5 -n 1;
    if [ $? = 0 ] 
    then
        echo "will overwrite"
    else
        echo "no response, terminating"
        exit 1
    fi
else
    mkdir -p cache
    echo "'cache' directory created"
fi

echo "downloading ciao from source"
cd cache
wget https://www.cse.msu.edu/~tangjili/datasetcode/ciao.zip
unzip ciao.zip; rm ciao.zip; mv ciao ciao_raw
cd ..

echo "downloading epinions from source"
cd cache
wget http://deepyeti.ucsd.edu/jmcauley/datasets/epinions/epinions_data.tar.gz 
tar -xzvf epinions_data.tar.gz; rm epinions_data.tar.gz; mv epinions_data epinions_raw
cd ..
