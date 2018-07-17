#!/bin/sh

EXTERNAL_IP=$1

# For TensorBoard
ssh -N -f -L localhost:16006:localhost:6006 zamlz@${EXTERNAL_IP}
# For Jupyter
ssh -N -f -L localhost:18888:localhost:8888 zamlz@${EXTERNAL_IP}
