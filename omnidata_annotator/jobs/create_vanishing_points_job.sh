#!/usr/bin/env bash

# Passed arguments : model_path {args}*

SCRIPT_PATH=${PWD}/scripts
model_path=$1

cd $model_path

# Get additional arguments
args="${@:2}"

# Generate vanishing points

blender -b --enable-autoexec -noaudio --python $SCRIPT_PATH/create_vanishing_points.py -- \
MODEL_PATH=$model_path $args