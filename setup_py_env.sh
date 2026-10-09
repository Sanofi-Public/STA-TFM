#!/bin/bash
eval "$(conda shell.bash hook)"

set -e

PY_ENV_PATH=$1
echo "[INFO] Using conda environment at: $PY_ENV_PATH"

# Create Python environment
if ! conda info --envs | grep -q "$PY_ENV_PATH"; then
    echo "Environment $PY_ENV_PATH does not exist. Creating..."
    conda create --override-channels -c conda-forge --prefix $PY_ENV_PATH python=3.10 -y
fi

# Activate python environment for installing dependencies
conda activate $PY_ENV_PATH
pip install -r requirements.txt

# Create the checkpoints directory if it does not exist
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
if [ ! -d "$CHECKPOINTS_DIR" ]; then
  echo "Creating checkpoints directory: $CHECKPOINTS_DIR"
  mkdir -p "$CHECKPOINTS_DIR"
fi


# Check if mmpose is installed in the specified environment
if ! python -c "import mmpose" &>/dev/null; then
    echo "mmpose and its dependecies are not installed in $PY_ENV_PATH. Cloning and installing..."
    pip install --no-input -r requirements.txt
    pip install mmengine

    # No mmcv wheels exist for torch>2.6: build from source with a CUDA toolkit matching torch
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
    conda install --override-channels -c conda-forge -y "cuda-version=$TORCH_CUDA" cuda-toolkit
    export CUDA_HOME="$CONDA_PREFIX"
    MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v --no-build-isolation "git+https://github.com/open-mmlab/mmcv.git@main"

    # Upstream MMDet pinned to main HEAD, patched to accept mmcv 2.2.0
    if [ ! -d "mmdetection_SDRD" ]; then
        git clone 'https://github.com/open-mmlab/mmdetection.git' mmdetection_SDRD
        git -C mmdetection_SDRD checkout cfd5d3a985b0249de009b67d04f37263e11cdf3d
        git -C mmdetection_SDRD apply ../patches/mmdet-mmcv-max-version.patch
        pip install -v --no-build-isolation ./mmdetection_SDRD
    fi

    # Clone and install mmpose only if the directory doesn't exist
    if [ ! -d "mmpose_SDRD" ]; then
        git clone 'https://github.com/open-mmlab/mmpose.git' mmpose_SDRD
        cd mmpose_SDRD
        pip install --no-cache-dir "setuptools<70" wheel "numpy<2"
        pip install --no-cache-dir --no-build-isolation chumpy
        pip install -r requirements.txt
        pip install -v --no-build-isolation .
        cd ..
    fi

    pip install --no-cache-dir "cython>=0.29"
    pip install --no-binary=xtcocotools --no-build-isolation --force-reinstall --no-cache-dir xtcocotools
    pip install --no-cache-dir --force-reinstall "setuptools<70"

else
    echo "mmpose is already installed in $PY_ENV_PATH. Skipping installation."
fi

# Optional: open a new shell in the environment
bash --rcfile <(echo "source $(conda info --base)/etc/profile.d/conda.sh; conda activate $PY_ENV_PATH")
