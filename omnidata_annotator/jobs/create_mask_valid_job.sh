#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
rm -rf mask_valid

# Make it if it doesn't exist
mkdir -p mask_valid

# Get additional arguments
args="${@:2}"

# Generate valid masks
# python $SCRIPT_PATH/create_mask_valid.py -- MODEL_PATH=$model_path $args

blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_mask_valid.py -- \
MODEL_PATH=$model_path $args