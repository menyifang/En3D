#!/bin/bash
MODEL=models/model_human.pkl
python lib/3DGEN/demo.py --outdir=res_seed --trunc=0.7 --seeds=0-10 \
--network=$MODEL --camera=models/camera.json --planes=False --type=human

