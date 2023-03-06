#!/bin/bash

python data/build_data.py --dset ciao 
python data/build_data.py --dset epinions
python data/show_statistics.py

