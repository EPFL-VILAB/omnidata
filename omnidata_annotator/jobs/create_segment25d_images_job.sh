#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
rm -rf segment_unsup25d

# Make it if it doesn't exist
mkdir -p segment_unsup25d

# Get additional arguments
args="${@:2}"

# Generate segment 2.5D images

blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_segmentation_25d_images.py -- \
MODEL_PATH=$model_path $args
