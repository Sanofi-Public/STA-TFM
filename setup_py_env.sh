#!/bin/bash
# Create the STA-TFM conda environment.
#
# Usage:
#   bash setup_py_env.sh <env-path>              # core: data generation, training, inference, evaluation
#   bash setup_py_env.sh <env-path> --with-demo  # + MMPose stack for the raw-video demo (Linux + CUDA GPU)
eval "$(conda shell.bash hook)"

set -e

PY_ENV_PATH=$1
WITH_DEMO=$2
if [ -z "$PY_ENV_PATH" ]; then
    echo "Usage: bash setup_py_env.sh <env-path> [--with-demo]"
    exit 1
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
echo "[INFO] Using conda environment at: $PY_ENV_PATH"

# Create Python environment
if [ ! -d "$PY_ENV_PATH" ]; then
    echo "Environment $PY_ENV_PATH does not exist. Creating..."
    conda create --override-channels -c conda-forge --prefix "$PY_ENV_PATH" python=3.10 -y
fi

# Activate python environment for installing dependencies
conda activate "$PY_ENV_PATH"
pip install -r requirements.txt

# Standard folders used throughout the README
mkdir -p datasets/dhp19 datasets/haa4d weights results

if [ "$WITH_DEMO" != "--with-demo" ]; then
    echo "[INFO] Core install complete. Re-run with --with-demo to install MMPose for the raw-video demo."
    echo "[INFO] Activate with: conda activate $PY_ENV_PATH"
    exit 0
fi

# Check if mmpose is installed in the specified environment
if ! python -c "import mmpose" &>/dev/null; then
    echo "mmpose and its dependencies are not installed in $PY_ENV_PATH. Cloning and installing..."
    pip install mmengine

    # No mmcv wheels exist for torch>2.6: build from source with a CUDA toolkit matching torch
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or '')")
    if [ -z "$TORCH_CUDA" ]; then
        echo "[ERROR] The installed torch build has no CUDA support; the MMPose demo stack requires a CUDA build of torch."
        exit 1
    fi
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

echo "[INFO] Install complete. Activate with: conda activate $PY_ENV_PATH"
