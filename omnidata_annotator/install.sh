#!/usr/bin/env bash

docker run --rm -v $PWD:/app/ ainaz99/multi-label-data:latest /bin/bash -c \
    "cmake /app/scripts/CMakeLists.txt;
    cd /app/scripts/;
    make"
