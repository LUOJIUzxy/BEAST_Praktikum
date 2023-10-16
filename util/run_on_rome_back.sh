#!/bin/bash

ssh testbed.cos.lrz.de srun --chdir "$PWD" --partition=rome "$@"
