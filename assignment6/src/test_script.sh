#!/bin/bash

# TODO@Students: Q2b) Adjust for NUMA effects, first touch, and thread pinning

if [ -z $1 ]
then
  echo "Usage: test_script.sh <mode>"
  echo "Mode: please specify whether to run part1 or part2."
  exit 255
fi

# This statement runs your program in line buffering mode, duplicating stdout to stdout and perf_data.txt.
# This makes it suitable for processing using CI.
case "$1" in
  "part1")
    stdbuf --output=L  ./assignment6_part_1 134217728 268435456 | tee perf_data1.txt
    exit ${PIPESTATUS[0]}
    ;;
  "part2")
    OMP_NUM_THREADS=1 stdbuf --output=L ./assignment6_part_2 268435456 1048576 | tee perf_data2.txt
    exit ${PIPESTATUS[0]}
    ;;
  *)
    echo "Unknown code version: $1"
    exit 255
    ;;
esac
