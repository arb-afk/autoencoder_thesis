#!/bin/bash
source ~/tf_wsl_venv/bin/activate
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(find "$SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name lib | paste -sd ':' -):$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
python /mnt/d/School/Thesis/tensorflow_version/check_gpu.py
