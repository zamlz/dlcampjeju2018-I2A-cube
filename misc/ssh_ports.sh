#!/bin/sh

VM='tfvm'
EXTERNAL_IP=$( gcloud compute instances describe ${VM} | grep 'natIP' | awk '{print $2}' )

echo "Connecting to ${EXTERNAL_IP}"

echo "For TensorBoard..."
ssh -N -f -L localhost:16006:localhost:6006 zamlz@${EXTERNAL_IP}
echo "For Jupyter..."
ssh -N -f -L localhost:18888:localhost:8888 zamlz@${EXTERNAL_IP}
