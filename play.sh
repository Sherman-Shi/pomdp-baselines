#!/bin/bash

# Set environment and algorithm variables
ENV="minigrid"
ENV_STYLE="MiniGrid-SimpleCrossingS11N5-v0"
ALGO="sacd"

# Run Python script
python policies/main.py --cfg "configs/pomdp/${ENV}/${ENV_STYLE}/rnn.yml" --algo "${ALGO}"
