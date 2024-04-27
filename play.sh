#!/bin/bash

# Set environment and algorithm variables
ENV="lunalander"
ENV_STYLE="v"
ALGO="sacd"

# Run Python script
python policies/main.py --cfg "configs/pomdp/${ENV}/${ENV_STYLE}/rnn.yml" --algo "${ALGO}"
