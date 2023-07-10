#!/bin/bash

set -e
set -o pipefail

rm -rf build
mkdir build
cd build

if [ -n "$1" ]; then
  cmake -DTORCHAIR_INSTALL_DST=$1 ..
else
  cmake ..
fi

make install_torchair -j8
