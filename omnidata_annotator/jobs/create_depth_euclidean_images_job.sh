#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
rm -rf depth_euclidean

# Make it if it doesn't
mkdir -p depth_euclidean

# Get additional arguments
args="${@:2}"

# Generate depth euclidean images
blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_depth_euclidean_images.py -- \
MODEL_PATH=$model_path $args

