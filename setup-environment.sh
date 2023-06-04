#!/bin/bash

CONDA_PREFIX=./venv

# Deactivate and remove existing environment
conda deactivate
conda env remove -p $CONDA_PREFIX

# Create new environment
conda env create --prefix $CONDA_PREFIX --file environment.yml
conda activate $CONDA_PREFIX
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "CUDNN_PATH=$(dirname "$(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")")" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:\$CUDNN_PATH/lib" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda deactivate

# Verify install:
conda activate $CONDA_PREFIX
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
conda deactivate
