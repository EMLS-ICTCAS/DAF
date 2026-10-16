#!/bin/bash

# Run DAF on all four benchmarks
# Set DATA_ROOT to the directory containing your datasets, e.g.:
#   export DATA_ROOT=/path/to/your/data

CUDA_VISIBLE_DEVICES=0 python main.py --config ./exps/daf_cifar.json
CUDA_VISIBLE_DEVICES=0 python main.py --config ./exps/daf_imagenetr.json
CUDA_VISIBLE_DEVICES=0 python main.py --config ./exps/daf_imageneta.json
CUDA_VISIBLE_DEVICES=0 python main.py --config ./exps/daf_objectnet.json
