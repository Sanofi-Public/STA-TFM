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

# Demo stack, pinned to exact revisions. No MMCV wheels exist for torch>2.6, so MMCV, MMDetection and
# MMPose are built from source at these commits (main branch as of the dates shown).
MMCV_REF=d2111bda571f3b3eb53954af6f9cff25f34e626f    # mmcv 2.2.0, main @ 2026-09-28
MMDET_REF=cfd5d3a985b0249de009b67d04f37263e11cdf3d   # mmdet main, patched to accept mmcv 2.2.0
MMPOSE_REF=759b39c13fea6ba094afc1fa932f51dc1b11cbf9  # mmpose 1.3.2, main @ 2025-08-04

# Check if mmpose is installed in the specified environment
if ! python -c "import mmpose" &>/dev/null; then
    echo "mmpose and its dependencies are not installed in $PY_ENV_PATH. Cloning and installing..."
    pip install -r requirements-demo.txt

    # Build MMCV from source with a CUDA toolkit matching torch
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or '')")
    if [ -z "$TORCH_CUDA" ]; then
        echo "[ERROR] The installed torch build has no CUDA support; the MMPose demo stack requires a CUDA build of torch."
        exit 1
    fi
    conda install --override-channels -c conda-forge -y "cuda-version=$TORCH_CUDA" cuda-toolkit
    export CUDA_HOME="$CONDA_PREFIX"
    MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v --no-build-isolation "git+https://github.com/open-mmlab/mmcv.git@${MMCV_REF}"

    # MMDetection, patched to accept mmcv 2.2.0
    if [ ! -d "mmdetection_SDRD" ]; then
        git clone 'https://github.com/open-mmlab/mmdetection.git' mmdetection_SDRD
    fi
    git -C mmdetection_SDRD checkout "$MMDET_REF"
    git -C mmdetection_SDRD apply --reverse --check ../patches/mmdet-mmcv-max-version.patch 2>/dev/null \
        || git -C mmdetection_SDRD apply ../patches/mmdet-mmcv-max-version.patch
    pip install -v --no-build-isolation ./mmdetection_SDRD

    # MMPose
    if [ ! -d "mmpose_SDRD" ]; then
        git clone 'https://github.com/open-mmlab/mmpose.git' mmpose_SDRD
    fi
    git -C mmpose_SDRD checkout "$MMPOSE_REF"
    pip install --no-cache-dir "setuptools==69.5.1" "wheel==0.48.0" "cython==3.3.0"
    pip install --no-cache-dir --no-build-isolation "chumpy==0.70"
    pip install -v --no-build-isolation ./mmpose_SDRD

    # xtcocotools must be compiled against the installed numpy (1.26); --no-deps keeps numpy untouched
    pip install --no-binary=xtcocotools --no-build-isolation --force-reinstall --no-deps --no-cache-dir "xtcocotools==1.14.3"

else
    echo "mmpose is already installed in $PY_ENV_PATH. Skipping installation."
fi

python -c "import numpy, mmcv, mmdet, mmpose; print(f'[INFO] numpy {numpy.__version__}, mmcv {mmcv.__version__}, mmdet {mmdet.__version__}, mmpose {mmpose.__version__}')"
echo "[INFO] Install complete. Activate with: conda activate $PY_ENV_PATH"
