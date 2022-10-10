#!/usr/bin/env bash

MODEL_NAME=${1:-nadaformer}

cat - | tr " " "_" | sed "s/./& /g" | sed "s/\s$//g" |\
 python order_stat_transform.py --split-incoming-line |\
 fairseq-interactive \
    ./data-bin/gutenberg-data/multilingual/ \
    --source-lang freq --target-lang multi \
    --path ./experiments/freq2multilingual/train/$MODEL_NAME/checkpoints/checkpoint_best.pt \
    --seed 1917 --beam 100 --max-tokens 10000 |\
 rg "^H-" | cut -f3 | tr -d " " | tr "_" " "
