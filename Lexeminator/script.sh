#!/bin/bash

# clone SciTools repo
curl http://builds.scitools.com/all_builds/b800/Understand/Understand-4.0.800-Linux-64bit.tgz | tar xz
# copy the scitools directory to /opt
cp -R ./scitools /opt/
# create user www-data
useradd www-data
# grant permissions
chown -R www-data /opt/scitools
# copy the license file - change the name in the license file according to the current user
cp users.txt /opt/scitools/conf/license/
# create SciTools dir in ~/.config
mkdir ~/.config
mkdir ~/.config/SciTools
# copy two config files
cp Und.conf ~/.config/SciTools/
cp Understand.conf ~/.config/SciTools/
# add path
export PATH=$PATH:/opt/scitools/bin/linux64/ >> ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/opt/scitools/bin/linux64/python/ >> ~/.bashrc
