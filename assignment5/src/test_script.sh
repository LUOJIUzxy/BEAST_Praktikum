#!/bin/bash

export OMP_PLACES=cores
export OMP_PROC_BIND=close

#export LIBOMPTARGET_DEVICE_RTL_DEBUG=0x5 
#export LIBOMPTARGET_INFO=20
#export OMP_DISPLAY_ENV=VERBOSE
#export GOMP_DEBUG=1
#export OMP_DEBUG=1

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
    OMP_TARGET_OFFLOAD=mandatory stdbuf --output=L ./assignment5  32 512 | tee gpu.txt
    ;;
  "c")
    OMP_TARGET_OFFLOAD=disabled stdbuf --output=L ./assignment5 32 512 | tee cpu.txt
    ;;
  "all")
    bash "$0" g || exit 255
    bash "$0" c || exit 255
    ;;
  *)
    echo "Unknown code version: $1"
esac
	
