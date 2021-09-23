[![Coverage Status](https://coveralls.io/repos/github/sjanssen2/ggmap/badge.svg?branch=master)](https://coveralls.io/github/sjanssen2/ggmap?branch=master)
[![Build Status](https://travis-ci.org/sjanssen2/ggmap.svg?branch=master)](https://travis-ci.org/sjanssen2/ggmap)

# ggmap
ggmap is coded in Python 3.x

## Introduction
ggmap shall convert MetaPhlAn profiles into GreenGenes OTU based profiles.

## Install
 1. Clone github repo via: `git clone https://github.com/sjanssen2/ggmap.git`
 2. cd into the new directory `cd ggmap`
 3. install modules from sources `python setup.py develop --user`

## Use
Open the jupyter notebook convert_profiles.ipynb and execute all cells. It will convert the six MetaPhlAn profiles from the "examples" directory and converts them into one OTU table with the 97% GreenGenes OTU clusters.

## use in JupyterHub@BCF
 1. Create a new conda environment (say `notebookServer`) and install ggmap in this environment
 2. activate environment: `conda activate notebookServer`
 3. register environment kernel to jupyter: `python -m ipykernel install --user --name notebookServer --display-name "notebookServer"
 4. for qsub: create a file `~/.bash_profile` into which you copy and paste the conda init lines from your `~/.bashrc` file.
