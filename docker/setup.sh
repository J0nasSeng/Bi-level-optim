#!/bin/sh
pip uninstall -y torch
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
bash