#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
# rm -rf point_info
# rm -rf nonfixated

# Make it if it doesn't
mkdir -p point_info
mkdir -p nonfixated

# Get additional arguments
args="${@:2}"

# Generate correspondences

blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/generate_points.py -- \
MODEL_PATH=$model_path $args
