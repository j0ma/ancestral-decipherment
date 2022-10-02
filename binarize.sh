#!/usr/bin/env bash

set -euo pipefail

# Preprocesses the data for decipherment-based word linkage experiments

[ $# -lt 2 ] && echo "Too few arguments!" && exit 1

FOLDER=$1
BIN_FOLDER=$2
SRC_LANG=${3:-yid}
TGT_LANG=${4:-deu}
N_WORKERS=${5:-1}

mkdir -p "$FOLDER"
mkdir -p "$BIN_FOLDER"

# Step 1: Binarize data
mkdir -p "$BIN_FOLDER"
fairseq-preprocess \
    --source-lang "$SRC_LANG" \
    --target-lang "$TGT_LANG" \
    --trainpref "$FOLDER/train" \
    --validpref "$FOLDER/dev" \
    --testpref "$FOLDER/test" \
    --destdir "$BIN_FOLDER" \
    --workers "$N_WORKERS"
