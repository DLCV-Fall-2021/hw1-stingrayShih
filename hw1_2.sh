#!/bin/bash
wget -O 'hw1_2models.zip' 'https://www.dropbox.com/s/773cdafq9tjkab7/hw1_2models.zip?dl=1'
unzip hw1_2models.zip
python3 hw1_2eval.py --img_dir=$1 --save_dir=$2