[![Coverage Status](https://coveralls.io/repos/github/sjanssen2/ggmap/badge.svg?branch=master)](https://coveralls.io/github/sjanssen2/ggmap?branch=master)
[![Build Status](https://travis-ci.org/sjanssen2/ggmap.svg?branch=master)](https://travis-ci.org/sjanssen2/ggmap)

# ggmap
ggmap is coded in Python 3.x

## Introduction
ggmap shall convert MetaPhlAn profiles into GreenGenes OTU based profiles.

## Install
 0. install miniconda3: https://docs.conda.io/en/latest/miniconda.html
 1. create a dedicated conda environment: `conda create --name ggmap`
 2. activate new conda environment: `conda activate ggmap`
 3. clone github repo via: `git clone https://github.com/sjanssen2/ggmap.git`
 4. cd into the new directory `cd ggmap`
 5. install modules from sources `python setup.py develop --user`
     * should the above command fail, you can alternatively try to install dependencies via conda like ``conda install -c conda-forge `cat ci/conda_requirements.txt | cut -d ">" -f 1 | xargs` `` and thereafter repeat the command of step 3. 
     * only for BCF System: you probably have to set the proxy to enable conda to speak to the internet: `export ftp_proxy="http://proxy.computational.bio.uni-giessen.de:3128" && export http_proxy="http://proxy.computational.bio.uni-giessen.de:3128" && export https_proxy="http://proxy.computational.bio.uni-giessen.de:3128"`

### Install for JupyterHub (skip if you install on your local machine)
 6. install neccessary additional conda packages `conda install ipykernel ipython_genutils`
 7. make new kernel known to the hub: `python -m ipykernel install --user --name ggmap --display-name "ggmap"`

### Configure
 After the first use, ggmap will create a file called `.ggmaprc` in your home directory, (look at the content via `cat $HOME/.ggmaprc`). Through this file, you can set some default to save typing in the python function calls like conda environment names.
 
 8. I assume you already installed qiime2 (https://docs.qiime2.org/2021.8/install/), edit your `~/.ggmaprc` to replace an potentially outdated qiime2 environment name with the one you installed (in our example 2021.8). There is a row starting with `condaenv_qiime2: `, replace the given name with your actual one.
 9. If you are going to use a cluster to execute jobs (default), you need to create a directory: `mkdir $HOME/TMP` 
 10. ggmap needs to know the location of your miniconda3 prefix. This is typically located in $HOME/miniconda3. However, in the BCF system, we encuraged people to install it in the prefix $HOME/no_backup/miniconda3 (to avoid flooding our backup with millions of unimportant files). You need to adapt the `dir_conda: ` entry in your `~/.ggmaprc` file accordingly.
 
### Tests
#### Challenge 1: load python code
Create a new jupyter notebook with the ggmap kernel and type the following two lines in a cell:
```
   from ggmap.snippets import *
   from ggmap.analyses import *
```
If the cell produces output in a red box like, you managed to successfully load my code. Congratulations!
```
/homes/sjanssen/miniconda3/envs/notebookServer/lib/python3.7/site-packages/skbio/util/_testing.py:15: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
  import pandas.util.testing as pdt
ggmap is custome code from Stefan Janssen, download at https://github.com/sjanssen2/ggmap
Reading settings file '/homes/sjanssen/.ggmaprc'
```
#### Challenge 2: locally execute wrapped Qiime2 code
Create a dummy feature table like
```
counts = pd.DataFrame([{'sample': "sample.A", 'bact1': 10, 'bact2': 7, 'bact3': 0}, 
                       {'sample': "sample.B", 'bact1':  5, 'bact2': 3, 'bact3': 8},
                       {'sample': "sample.C", 'bact1': 10, 'bact2': 0, 'bact3': 1}]).set_index('sample').T
```                       
Use this feature table to compute beta diversity distances through one of the wrapper functions of ggmap that internally call qiime2 methods:
`res = beta_diversity(counts, metrics=['jaccard'], dry=False, use_grid=False)`
Should it run through, you should "see" a result like the following when executing `res['results']['jaccard']` in a new cell:
![image](https://user-images.githubusercontent.com/11960616/134654180-17892128-8258-45a4-b6c3-7d51fc933bee.png)

#### Challenge 3: use SGE/Slurm to execute wrapped Qiime2 code
As above, but now we want to distribute computation as a cluster job via `res = beta_diversity(counts, metrics=['jaccard'], dry=False, use_grid=True, nocache=True)`

Result should be the same as above, but the system should submit the job to the SGE grid engine and poll every 10 seconds for the result. You might want to use another terminal and observe the job status via `qstat` and/or look into the sub-directory `$HOME/TMP/`.

*Good luck!*
