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
  TESTCASE_FILTER="*"
  # Process the options
  while getopts 'hj:vuisck:g:' opt
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
      k) TESTCASE_FILTER=$OPTARG ;;
      i) ENABLE_CI_BUILD_AND_INSTALL="on" ;;
      *) logging "Undefined option: ${opt}"
         usage
         exit 1 ;;
    esac
  done
}

build_torchair() {
  local BUILD_PATH="${TORCHAIR_ROOT}/build/compile"
  rm -rf "${BUILD_PATH}"
  mkdir -pv "${BUILD_PATH}"
  cp -r "${TORCHAIR_ROOT}/python" "${BUILD_PATH}"

  local RELEASE_PATH="${TORCHAIR_ROOT}/output"
  mkdir -pv "${RELEASE_PATH}"
  cd "${BUILD_PATH}/python" && python3 setup.py bdist_wheel --dist-dir ${RELEASE_PATH}
}

install_torchair() {
  local RELEASE_PATH="${TORCHAIR_ROOT}/output"
  pip3 uninstall npu_extension_for_inductor -y
  pip3 install ${RELEASE_PATH}/*.whl
}

run_test() {
  local TYPE="$1"
  local TEST_PATH="${TORCHAIR_ROOT}/build/${TYPE}"
  rm -rf "${TEST_PATH}"
  mkdir -pv "${TEST_PATH}"
  cp -r "${TORCHAIR_ROOT}/python" "${TEST_PATH}"
  export PYTHONPATH="${TEST_PATH}/python"
  cd "${TEST_PATH}"

  export ASCIR_NOT_READY=1
  export NPU_INDUCTOR_FALLBACK_INT64=0
  if [[ "X$TYPE" = "Xut" ]]; then
    ${PYTHON_BIN_PATH} -m unittest discover -s "${TORCHAIR_ROOT}/tests" -p "test_*.py" -v -k "${TESTCASE_FILTER}"
  elif [[ "X$TYPE" = "Xst" ]]; then
    ${PYTHON_BIN_PATH} -m unittest discover -s "${TORCHAIR_ROOT}/tests" -p "test_*.py" -v -k "${TESTCASE_FILTER}"
  fi
  mkdir -pv "${TORCHAIR_ROOT}/coverage"
  lcov --capture --initial --directory . --output-file "${TORCHAIR_ROOT}/coverage/coverage.info"
}

main() {
  checkopts "$@"

  PYTHON_BIN_PATH=$(which python3.8 || which python3)
  export TARGET_PYTHON_PATH=${PYTHON_BIN_PATH}
  PYTHON_BIN_PATH=${PYTHON_BIN_PATH} bash ${TORCHAIR_ROOT}/configure

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
