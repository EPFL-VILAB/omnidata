#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
rm -rf rgb

# Make it if it doesn't exist
mkdir -p rgb

# Get additional arguments
args="${@:2}"

# Generate normal images

blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_rgb_images_textured.py -- \
MODEL_PATH=$model_path $args
