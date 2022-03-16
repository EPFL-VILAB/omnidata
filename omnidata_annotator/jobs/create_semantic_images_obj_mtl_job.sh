#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
rm -rf semantic

# Make it if it doesn't exist
mkdir -p semantic

# Get additional arguments
args="${@:2}"

# Generate semantic segmentation images
blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_semantic_images_obj_mtl.py -- \
MODEL_PATH=$model_path $args