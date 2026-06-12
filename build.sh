#!/bin/bash
set -e   # stop immediately if the compile fails
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  -undefined dynamic_lookup \
  $(python3 -m pybind11 --includes) \
  src/sampler.cpp src/bindings.cpp \
  -o csampler$(python3-config --extension-suffix)
echo "built csampler"