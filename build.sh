#!/bin/bash

set -e
set -o pipefail

TORCHAIR_ROOT=$(cd "$(dirname $0)"; pwd)

# print usage message
usage() {
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-g] [-u] [-s] [-c]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -j[n] Set the number of threads used to build CANN, default is 8"
  echo "    -v Verbose"
  echo "    -g GCC compiler prefix, used to specify the compiler toolchain"
  echo "    -u torchair utest"
  echo "    -s torchair stest"
  echo "    -c torchair ci build"
  echo "    -i torchair ci build and install"
  echo "to be continued ..."
}

logging() {
  echo "[INFO] $@"
}

# parse and set optionss
checkopts() {
  VERBOSE=""
  THREAD_NUM=8
  GCC_PREFIX=""
  ENABLE_TORCHAIR_UT="off"
  ENABLE_TORCHAIR_ST="off"
  ENABLE_CI_BUILD="off"
  # Process the options
  while getopts 'hj:vuiscg:' opt
  do
    case "${opt}" in
      h) usage
         exit 0 ;;
      j) THREAD_NUM=$OPTARG ;;
      v) VERBOSE="VERBOSE=1" ;;
      g) GCC_PREFIX=$OPTARG ;;
      u) ENABLE_TORCHAIR_UT="on" ;;
      s) ENABLE_TORCHAIR_ST="on" ;;
      c) ENABLE_CI_BUILD="on" ;;
      i) ENABLE_CI_BUILD_AND_INSTALL="on" ;;
      *) logging "Undefined option: ${opt}"
         usage
         exit 1 ;;
    esac
  done
}

PACKAGE_NAME=torchair-0.1-py3-none-any.whl

install_submodules(){
  logging "--- Trying to initialize submodules"
  git submodule update --init --recursive
}

build_torchair() {
  logging "CMake Args: ${CMAKE_ARGS}"

  local CMAKE_PATH="${TORCHAIR_ROOT}/build"
  mkdir -pv "${CMAKE_PATH}"
  cd "${CMAKE_PATH}" && cmake ${CMAKE_ARGS} ..

  make torchair ${VERBOSE} -j${THREAD_NUM}

  local RELEASE_PATH="${TORCHAIR_ROOT}/output"
  mkdir -pv "${RELEASE_PATH}"
  mv "${TORCHAIR_ROOT}/build/dist/dist/${PACKAGE_NAME}" ${RELEASE_PATH}
}

install_torchair() {
  local RELEASE_PATH="${TORCHAIR_ROOT}/output"
  pip3 uninstall torchair -y
  pip3 install ${RELEASE_PATH}/${PACKAGE_NAME}
}

run_test() {
  local TYPE="$1"

  local CMAKE_PATH="${TORCHAIR_ROOT}/tests/build"
  mkdir -pv "${CMAKE_PATH}"
  cd "${CMAKE_PATH}" && cmake .. -DCMAKE_BUILD_TYPE=GCOV -DTORCHAIR_INSTALL_DST=${CMAKE_PATH}/${TYPE}/torchair -DPYTHON_BIN_PATH=${PYTHON_BIN_PATH}

  export PYTHONPATH=${CMAKE_PATH}/${TYPE}:${PYTHONPATH}
  export LD_LIBRARY_PATH=${CMAKE_PATH}/stubs:${ASCEND_SDK_PATH}/lib:${ASCEND_SDK_PATH}/lib64:${LD_LIBRARY_PATH}

  mkdir -pv "${TORCHAIR_ROOT}/coverage"
  make torchair_${TYPE} -j${THREAD_NUM}
  lcov -o ${TORCHAIR_ROOT}/coverage/coverage.info -e ${CMAKE_PATH}/${TYPE}/${TYPE}.coverage "*torchair/torchair*"
}

main() {
  checkopts "$@"

  if [[ "$TARGET_PYTHON_PATH" == "" ]]; then
      PYTHON_BIN_PATH=$(which python3 || which python3.8)
      export TARGET_PYTHON_PATH=${PYTHON_BIN_PATH}
  fi
  if [[ "X$ASCEND_CUSTOM_PATH" = "X" ]]; then
    if [[ "X$ENABLE_CI_BUILD" = "Xon" || "X$ENABLE_CI_BUILD_AND_INSTALL" = "Xon" ]]; then
      echo "Building torchair with no ascned-sdk specified"
      export NO_ASCEND_SDK=1
    else
      if [[ "X$ASCEND_HOME_PATH" = "X" ]]; then
        echo "ASCEND_CUSTOM_PATH or ASCEND_HOME_PATH must be set when running ut or st"
        exit 1
      else
        echo "Building torchair with ascned-sdk in ${ASCEND_HOME_PATH}"
        export ASCEND_SDK_PATH=${ASCEND_HOME_PATH}
      fi
    fi
  else
    echo "Building torchair with ascned-sdk in ${ASCEND_CUSTOM_PATH}"
    export ASCEND_SDK_PATH=${ASCEND_CUSTOM_PATH}/opensdk/opensdk/
  fi
  PYTHON_BIN_PATH=${PYTHON_BIN_PATH}
  bash ${TORCHAIR_ROOT}/configure

  ${GCC_PREFIX}g++ -v

  if [[ "$GCC_PREFIX" != "" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGCC_PREFIX=$GCC_PREFIX"
  fi

  install_submodules

  if [[ "X$ENABLE_CI_BUILD" = "Xon" ]]; then
    build_torchair
  fi

  if [[ "X$ENABLE_CI_BUILD_AND_INSTALL" = "Xon" ]]; then
    build_torchair
    install_torchair
  fi

  if [[ "X$ENABLE_TORCHAIR_UT" = "Xon" ]]; then
    run_test "ut"
  fi

  if [[ "X$ENABLE_TORCHAIR_ST" = "Xon" ]]; then
    run_test "st"
  fi
}

main "$@"
