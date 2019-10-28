# mv -v $CONDA_PREFIX/x86_64-conda_cos6-linux-gnu/bin/ld $CONDA_PREFIX/x86_64-conda_cos6-linux-gnu/bin/ld_bin
# echo "#!/usr/bin/bash" > $CONDA_PREFIX/x86_64-conda_cos6-linux-gnu/bin/ld
# echo "ld -L/usr/lib64/ -L$CONDA_PREFIX/x86_64-conda_cos6-linux-gnu/lib/ $@" >> $CONDA_PREFIX/x86_64-conda_cos6-linux-gnu/bin/ld
# chmod a+x $CONDA_PREFIX/x86_64-conda_cos6-linux-gnu/bin/ld

echo "install.packages(c('devtools'), repos='http://cran.us.r-project.org')" > install.r
echo "library('devtools')" >> install.r
echo "Sys.setenv(TAR='/bin/tar')" >> install.r
echo "devtools::install_github('benjjneb/dada2', ref='v1.12', upgrade_dependencies=FALSE)" >> install.r
echo "if (!requireNamespace('BiocManager', quietly = TRUE))" >> install.r
echo "  install.packages('BiocManager', repos='http://cran.us.r-project.org')" >> install.r
echo "BiocManager::install('gridExtra', upgrade_dependencies=FALSE)" >> install.r
echo "BiocManager::install('phyloseq', upgrade_dependencies=FALSE)" >> install.r
cat install.r | R --vanilla
