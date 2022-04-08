##!/usr/bin/env bash

# wget https://drive.switch.ch/index.php/s/RFfTZwyKROKKx0l/download
# unzip -j download -d pretrained_models
# rm download

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get install apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update -y && sudo apt-get install -y google-cloud-sdk

sudo apt install -y imagemagick

pip install gdown
mkdir -p pretrained_models 

gdown https://drive.google.com/uc?id=1iJjV9rkdeLvsTU9x3Vx8vwZUg-sSQ9nm -O ./pretrained_models/ # omnidata normals (v1)
gdown https://drive.google.com/uc?id=1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR -O ./pretrained_models/ # omnidata normals (v2)
