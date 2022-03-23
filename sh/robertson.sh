#!/bin/sh

python3 Radiance_generate.py -s Photos/JPG -d RobertsonData -m robertson
python3 HDRI_generate.py -b RobertsonData/Radiance_B.npy -g RobertsonData/Radiance_G.npy -r RobertsonData/Radiance_R.npy -d RobertsonData
python3 HDRI_transform.py -b RobertsonData/Radiance_B.npy -g RobertsonData/Radiance_G.npy -r RobertsonData/Radiance_R.npy -m global -d RobertsonData