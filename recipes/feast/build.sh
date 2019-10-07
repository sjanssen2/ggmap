echo "install.packages(c('ggrepel'), repos='http://cran.us.r-project.org')" | R --vanilla
mkdir -p $PREFIX/src/feast
cp 'FEAST_src/FEAST_ main.R' $PREFIX/src/feast/feast_main.R
cp 'FEAST_src/src.R' $PREFIX/src/feast/
