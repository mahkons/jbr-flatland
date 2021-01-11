#!/bin/bash

# python cython_setup.py build_ext --inplace

# if [[ "$1" == "local" ]]; then
    # python train.py # ./run_local.py ?
# else
    # python run_remote.py
# fi

# rm -rf env/observations/SimpleObservation.c
# rm -rf env/observations/TreeObsForRailEnv.c

python run_remote.py
