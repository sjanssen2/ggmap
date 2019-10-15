mkdir -p $CONDA_PREFIX/src/bugbase
cp -v -r * $CONDA_PREFIX/src/bugbase/

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "#!/bin/bash" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export PATH=$CONDA_PREFIX/src/bugbase/bin:$PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export BUGBASE_PATH=$CONDA_PREFIX/src/bugbase" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

$CONDA_PREFIX/src/bugbase/bin/run.bugbase.r -h
