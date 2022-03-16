#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Clear the directory if it exists
rm -rf keypoints2d

# Make it if it doesn't exist
mkdir -p keypoints2d

# Get additional arguments
args="${@:2}"

# Generate 2d keypoint images
blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_keypoints_2d_images.py -- \
MODEL_PATH=$model_path $args

# python $SCRIPT_PATH/create_keypoints_2d_images.py -- MODEL_PATH=$model_path $args

