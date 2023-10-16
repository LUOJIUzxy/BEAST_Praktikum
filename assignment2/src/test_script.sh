#!/bin/bash

# TODO@Students: Part I Q2b) Adjust for NUMA effects, first touch, and thread pinning

if [ -z $1 ]
then
  echo "Usage: test_script.sh <g|c|all>"
  echo "Mode: please specify whether to run on gpu(g) or cpu(c) or both(all)."
  exit 255
fi

# This statement runs your program in line buffering mode, duplicating stdout to stdout and perf_data.txt.
# This makes it suitable for processing using CI.
case "$1" in
  "g")
    OMP_TARGET_OFFLOAD=mandatory stdbuf --output=L ./assignment2 134217728 268435456 | tee gpu.txt
    ;;
  "c")
    OMP_TARGET_OFFLOAD=disabled stdbuf --output=L ./assignment2 134217728 268435456 | tee cpu.txt
    ;;
  "all")
    bash "$0" g || exit 255
    bash "$0" c || exit 255
    ;;
  *)
    echo "Unknown code version: $1"
esac

	
