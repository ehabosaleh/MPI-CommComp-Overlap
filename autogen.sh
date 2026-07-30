#!/bin/sh

rm -rf build
cmake -S . -B build
cmake --build build
cmake --install build --prefix $PWD

rm -rf build
