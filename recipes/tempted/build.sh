# avoid 'libxml/globals.h: No such file or directory' when installing phyloseq through Bioconductor
echo 'install.packages("XML", repos="http://cran.us.r-project.org")' | R --vanilla
ln -s $CONDA_PREFIX/include/libxml2 $CONDA_PREFIX/lib/R/include/libxml2
# when installing microTensor, igraph get's compiled again?! but this time, the headers are expected to be in the following path
ln -s $CONDA_PREFIX/include/libxml2/libxml $CONDA_PREFIX/lib/R/include/libxml

# install bioconductor and other R packages into new conda env
echo 'install.packages(c("BiocManager","IRkernel", "remotes"), repos="http://cran.us.r-project.org")' | R --vanilla

echo 'BiocManager::install(c("phyloseq", "PERMANOVA"), update=TRUE, ask=FALSE)' | R --vanilla

# install libraries from github repos
echo 'remotes::install_github("syma-research/microTensor")' | R --vanilla
echo 'remotes::install_github("jbisanz/qiime2R")' | R --vanilla
echo 'remotes::install_github("pixushi/tempted")' | R --vanilla
