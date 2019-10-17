echo "install.packages(c('devtools'), repos='http://cran.us.r-project.org')" > install.r
echo "library('devtools')" >> install.r
echo "if (!requireNamespace('BiocManager', quietly = TRUE))" >> install.r
echo "  install.packages('BiocManager', repos='http://cran.us.r-project.org')" >> install.r
echo "BiocManager::install('DESeq2')" >> install.r
echo "BiocManager::install('metagenomeSeq')" >> install.r
echo "BiocManager::install('edgeR')" >> install.r
echo "install.packages('MetaLonDA', repos='http://cran.us.r-project.org')" >> install.r
cat install.r | R --vanilla
