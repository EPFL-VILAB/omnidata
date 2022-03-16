#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
rm -rf depth_zbuffer

# Make it if it doesn't exist
mkdir -p depth_zbuffer

# Get additional arguments
args="${@:2}"

# Generate correspondences

blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_depth_zbuffer_images.py -- \
MODEL_PATH=$model_path $args