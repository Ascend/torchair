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
  while getopts 'hj:vuscg:' opt
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
      *) logging "Undefined option: ${opt}"
         usage
         exit 1 ;;
    esac
  done
}

build_torchair() {
  cd "${TORCHAIR_ROOT}" && bash ./configure

  logging "CMake Args: ${CMAKE_ARGS}"

  local CMAKE_PATH="${TORCHAIR_ROOT}/build"
  mkdir -pv "${CMAKE_PATH}"
  cd "${CMAKE_PATH}" && cmake ${CMAKE_ARGS} ..

  make torchair ${VERBOSE} -j${THREAD_NUM}

  local RELEASE_PATH="${TORCHAIR_ROOT}/output"
  mkdir -pv "${RELEASE_PATH}"
  mv "${TORCHAIR_ROOT}/build/dist/dist/torchair-0.1-py3-none-any.whl" ${RELEASE_PATH}
}

run_test() {
  local TYPE="$1"

  local CMAKE_PATH="${TORCHAIR_ROOT}/tests/build"
  mkdir -pv "${CMAKE_PATH}"
  cd "${CMAKE_PATH}" && cmake .. -DCMAKE_BUILD_TYPE=GCOV -DTORCHAIR_INSTALL_DST=${CMAKE_PATH}/${TYPE}/torchair -DPYTHON_BIN_PATH=${PYTHON_BIN_PATH}

  export PYTHONPATH=${CMAKE_PATH}/${TYPE}:${PYTHONPATH}
  export LD_LIBRARY_PATH=${CMAKE_PATH}/stubs:${ASCEND_SDK_PATH}/lib:${LD_LIBRARY_PATH}

  mkdir -pv "${TORCHAIR_ROOT}/coverage"
  make torchair_${TYPE} -j${THREAD_NUM}
  lcov -o ${TORCHAIR_ROOT}/coverage/coverage.info -e ${CMAKE_PATH}/${TYPE}/${TYPE}.coverage "*torchair/torchair*"
}

main() {
  checkopts "$@"

  PYTHON_BIN_PATH=$(which python3)
  export TARGET_PYTHON_PATH=${PYTHON_BIN_PATH}
  if [[ "X$ASCEND_CUSTOM_PATH" = "X" ]]; then
    if [[ "X$ENABLE_CI_BUILD" = "Xon" ]]; then
      echo "Building torchair with no ascned-sdk specified"
      export NO_ASCEND_SDK=1
    else
      echo "ASCEND_CUSTOM_PATH must be set when running ut or st"
      exit 1
    fi
  else
    echo "Building torchair with ascned-sdk in ${ASCEND_CUSTOM_PATH}"
    export ASCEND_SDK_PATH=${ASCEND_CUSTOM_PATH}/opensdk/opensdk/
  fi

  ${GCC_PREFIX}g++ -v

  if [[ "$GCC_PREFIX" != "" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DGCC_PREFIX=$GCC_PREFIX"
  fi

  if [[ "X$ENABLE_CI_BUILD" = "Xon" ]]; then
    build_torchair
  fi

  if [[ "X$ENABLE_TORCHAIR_UT" = "Xon" ]]; then
    run_test "ut"
  fi

  if [[ "X$ENABLE_TORCHAIR_ST" = "Xon" ]]; then
    run_test "st"
  fi
}

main "$@"
