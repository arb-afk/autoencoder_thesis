#!/bin/bash
# Convenience wrapper: activates the WSL2 GPU-enabled TensorFlow venv and
# runs the given command from the tensorflow_version directory.
# Usage: bash run_wsl.sh main.py --model all --max-users 20 --device gpu
set -e
source ~/tf_wsl_venv/bin/activate
cd /mnt/d/School/Thesis/tensorflow_version
python "$@"
