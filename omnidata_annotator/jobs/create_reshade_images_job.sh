#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
rm -rf reshading

# Make it if it doesn't exist
mkdir -p reshading

# Get additional arguments
args="${@:2}"

# Generate reshade images
blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_albedo_images.py -- \
MODEL_PATH=$model_path $args
