#!/bin/bash

ROME_WORK_DIR="$PWD"
ssh testbed.cos.lrz.de "srun --chdir '$ROME_WORK_DIR' --partition=rome --nodelist=rome2 ssh 127.0.0.1 'cd $ROME_WORK_DIR && $@'"
