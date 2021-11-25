#!/bin/bash
wget -O 'hw1_1models.zip' 'https://www.dropbox.com/s/gsdirwcv6fp9ls2/1_1.zip?dl=1'
unzip hw1_1models.zip
python3 hw1_1eval.py --img_dir=$1 --save_dir=$2
