#!/usr/bin/env bash

set -euo pipefail

# Command line arguments & defaults.
EXPERIMENT_NAME=$1
MODEL_NAME=${2:-transformer}
SPLIT=${3:-test}
BEAM=${4:-5}
SEED=${5:-1917}
SRC_LANG=${6:-yid}
TGT_LANG=${7:-deu}

EXPERIMENT_FOLDER="$(pwd)/experiments/${EXPERIMENT_NAME}"
TRAIN_FOLDER="${EXPERIMENT_FOLDER}/train/${MODEL_NAME}"
DATA_BIN_FOLDER="${TRAIN_FOLDER}/binarized_data"
CHECKPOINT_FOLDER="${TRAIN_FOLDER}/checkpoints"
RAW_DATA_FOLDER="${TRAIN_FOLDER}/raw_data"
EVAL_OUTPUT_FOLDER="${EXPERIMENT_FOLDER}/eval/eval_${MODEL_NAME}"

echo "DATA_BIN_FOLDER=${DATA_BIN_FOLDER}"
echo "CHECKPOINT_FOLDER=${CHECKPOINT_FOLDER}"
echo "SPLIT=${SPLIT}"
echo "BEAM=${BEAM}"
echo "SEED=${SEED}"
echo "RAW_DATA_FOLDER=${RAW_DATA_FOLDER}"

# Prediction options.

evaluate() {
    local -r DATA_BIN_FOLDER="$1"
    shift
    local -r EXPERIMENT_FOLDER="$1"
    shift
    local -r CHECKPOINT_FOLDER="$1"
    shift
    local -r SPLIT="$1"
    shift
    local -r BEAM_SIZE="$1"
    shift
    local -r SEED="$1"
    shift

    echo "seed = ${SEED}"

    # Checkpoint file
    CHECKPOINT_FILE="${CHECKPOINT_FOLDER}/checkpoint_best.pt"
    if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
        echo "${CHECKPOINT_FILE} not found. Changing..."
        CHECKPOINT_FILE="${CHECKPOINT_FILE/checkpoint_best/checkpoint_last}"
        echo "Changed checkpoint file to: ${CHECKPOINT_FILE}"
    fi

    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_SPLIT="${SPLIT/dev/valid}"

    OUT="${EVAL_OUTPUT_FOLDER}/${SPLIT}.out"
    SOURCE_TSV="${EVAL_OUTPUT_FOLDER}/${SPLIT}_with_source.tsv"
    SOURCE_LANGS_TSV="${EVAL_OUTPUT_FOLDER}/${SPLIT}_with_source_and_langs.tsv"
    GOLD="${EVAL_OUTPUT_FOLDER}/${SPLIT}.gold"
    HYPS="${EVAL_OUTPUT_FOLDER}/${SPLIT}.hyps"
    SOURCE="${EVAL_OUTPUT_FOLDER}/${SPLIT}.source"
    LANGS="${EVAL_OUTPUT_FOLDER}/${SPLIT}.languages"
    SCORE="${EVAL_OUTPUT_FOLDER}/${SPLIT}.eval.score"
    SCORE_TSV="${EVAL_OUTPUT_FOLDER}/${SPLIT}_eval_results.tsv"

    if [ -z "${CUDA_VISIBLE_DEVICES}" ]
    then
        CPU_FLAG="--cpu"
    else
        CPU_FLAG=""
    fi

    echo "Evaluating into ${OUT}"

        #"${CPU_FLAG}" \
    # Make raw predictions
    fairseq-generate \
        "${DATA_BIN_FOLDER}" \
        --source-lang="${SRC_LANG}" \
        --target-lang="${TGT_LANG}" \
        --path="${CHECKPOINT_FILE}" \
        --seed="${SEED}" \
        --gen-subset="${FAIRSEQ_SPLIT}" \
        --beam="${BEAM_SIZE}" \
        --max-tokens 10000 \
        --no-progress-bar | tee "${OUT}"

    # Also separate gold/system output/source into separate text files
    # (Sort by index to ensure output is in the same order as plain text data)
    cat "${OUT}" | grep '^T-' | sed "s/^T-//g" | sort -k1 -n | cut -f2 >"${GOLD}"
    cat "${OUT}" | grep '^H-' | sed "s/^H-//g" | sort -k1 -n | cut -f3 >"${HYPS}"
    cat "${OUT}" | grep '^S-' | sed "s/^S-//g" | sort -k1 -n | cut -f2 >"${SOURCE}"

    paste "${GOLD}" "${HYPS}" "${SOURCE}" >"${SOURCE_TSV}"

    # If using multiple languages, this will need to change.
    cat "${SOURCE_TSV}" | while read line; do echo "${TGT_LANG}"; done >"${LANGS}"

    paste "${SOURCE_TSV}" "${LANGS}" >"${SOURCE_LANGS_TSV}"

    python evaluate.py \
        --tsv "${SOURCE_LANGS_TSV}" \
        --score-output-path "${SCORE_TSV}" \
        --output-as-tsv

    # Finally output the score so Guild.ai grab it
    cat "${SCORE_TSV}"
}

evaluate "${DATA_BIN_FOLDER}" "${EXPERIMENT_FOLDER}" "${CHECKPOINT_FOLDER}" "${SPLIT}" "${BEAM}" "${SEED}"
