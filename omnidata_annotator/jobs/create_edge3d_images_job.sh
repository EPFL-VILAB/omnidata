#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
rm -rf edge_occlusion

# Make it if it doesn't exist
mkdir -p edge_occlusion

# Get additional arguments
args="${@:2}"

# Generate 3D edge images

blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_edge_3d_images.py -- \
MODEL_PATH=$model_path $args
