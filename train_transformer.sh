#!/usr/bin/env bash

DATA_BIN_PATH=$1
CHECKPOINT_FOLDER=$2
SRC_LANG=${3:-yid}
TGT_LANG=${4:-deu}

LR=0.0005
EED=56 #512
EHS=200 #2048
DED=56 # 512
DHS=200 #2048
SEED=1917
PATIENCE=5
P_DROPOUT=0.1
CRITERION=label_smoothed_cross_entropy
OPTIMIZER=adam
CLIP_NORM=1.0
MAX_TOKENS=1000 #0
MAX_UPDATE=1000 #00
LR_SCHEDULER=inverse_sqrt
ACTIVATION_FN=relu
SAVE_INTERVAL=1
ENCODER_LAYERS=6
DECODER_LAYERS=6
LABEL_SMOOTHING=0.1
ENCODER_ATTENTION_HEADS=8
DECODER_ATTENTION_HEADS=8
VALIDATE_INTERVAL=1
WARMUP_INIT_LR=0.001
WARMUP_UPDATES=100
MAX_EPOCHS=20

train() {
    local -r CP="$1"
    shift

    fairseq-train \
        "${DATA_BIN_PATH}" \
        --task translation \
        --cpu \
        --save-dir="${CP}" \
        --source-lang="${SRC_LANG}" \
        --target-lang="${TGT_LANG}" \
        --log-format="json" \
        --seed="${SEED}" \
        --patience=${PATIENCE} \
        --arch=transformer \
        --attention-dropout="${P_DROPOUT}" \
        --activation-dropout="${P_DROPOUT}" \
        --activation-fn="${ACTIVATION_FN}" \
        --encoder-embed-dim="${EED}" \
        --encoder-ffn-embed-dim="${EHS}" \
        --encoder-layers="${ENCODER_LAYERS}" \
        --encoder-attention-heads="${ENCODER_ATTENTION_HEADS}" \
        --encoder-normalize-before \
        --decoder-embed-dim="${DED}" \
        --decoder-ffn-embed-dim="${DHS}" \
        --decoder-layers="${DECODER_LAYERS}" \
        --decoder-attention-heads="${DECODER_ATTENTION_HEADS}" \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --criterion="${CRITERION}" \
        --label-smoothing="${LABEL_SMOOTHING}" \
        --optimizer="${OPTIMIZER}" \
        --lr="${LR}" \
        --lr-scheduler="${LR_SCHEDULER}" \
        --clip-norm="${CLIP_NORM}" \
        --max-tokens="${MAX_TOKENS}" \
        --save-interval="${SAVE_INTERVAL}" \
        --validate-interval="${VALIDATE_INTERVAL}" \
        --adam-betas '(0.9, 0.98)' --update-freq=16 \
        --no-epoch-checkpoints \
        --skip-invalid-size-inputs-valid-test \
        --warmup-updates "${WARMUP_UPDATES}" \
        --warmup-init-lr "${WARMUP_INIT_LR}" \
        --max-epoch "${MAX_EPOCHS}"
        #--max-update="${MAX_UPDATE}" \
        #--fp16 \

}

train "${CHECKPOINT_FOLDER}"
