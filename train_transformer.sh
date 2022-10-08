#!/usr/bin/env bash

EXPERIMENT_NAME=$1
MODEL_NAME=$2
SRC_LANG=${3:-yid}
TGT_LANG=${4:-deu}

EXPERIMENT_FOLDER="./experiments/${EXPERIMENT_NAME}"
TRAIN_FOLDER="${EXPERIMENT_FOLDER}/train/${MODEL_NAME}"
DATA_BIN_PATH="${TRAIN_FOLDER}/binarized_data"
CHECKPOINTS_PATH="${TRAIN_FOLDER}/checkpoints"
TENSORBOARD_PATH="${TRAIN_FOLDER}/tensorboard"

. transformer_config.sh

train() {
    local -r CP="$1"
    shift

    fairseq-train \
        "${DATA_BIN_PATH}" \
        --task translation \
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
        --adam-betas '(0.9, 0.98)' --update-freq=1 \
        --no-epoch-checkpoints \
        --skip-invalid-size-inputs-valid-test \
        --warmup-updates "${WARMUP_UPDATES}" \
        --warmup-init-lr "${WARMUP_INIT_LR}" \
        --tensorboard-logdir "${TENSORBOARD_PATH}" \
        --max-epoch "${MAX_EPOCHS}" --fp16 --reset-optimizer
}

train "${CHECKPOINTS_PATH}"
