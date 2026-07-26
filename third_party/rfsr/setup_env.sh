#!/usr/bin/env bash

# install required packages
sudo apt update
xargs sudo apt-get -y install < ubuntu_requirements.txt

# install python virtual environment
git submodule update --init --recursive
python3 -m venv venv/
source venv/bin/activate
python3 -m pip install -r requirements.txt
# add project to python path
export PYTHONPATH=$PYTHONPATH:`pwd`
