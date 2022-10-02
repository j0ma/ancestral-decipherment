#!/usr/bin/env bash

tsv_file=$1

panlex_unsplit_plain_folder=./_data/unsplit/panlex/plain
panlex_split_plain_folder=./_data/split/panlex/plain
panlex_split_space_separated_folder=./_data/split/panlex/space_separated
panlex_split_freq_encoded_folder=./_data/split/panlex/freq_encoded
panlex_split_freq2space_folder=./_data/split/panlex/freq2space

# Create parallel plain text (not split)
tail +2 "${tsv_file}" |
    sort | uniq |
    pee \
        "cut -f1 > ${panlex_unsplit_plain_folder}/plain.deu" \
        "cut -f2 > ${panlex_unsplit_plain_folder}/plain.yid"

# Split plaintext into train/dev/test
python split.py \
    --source-file "${panlex_unsplit_plain_folder}/plain.yid" \
    --target-file "${panlex_unsplit_plain_folder}/plain.deu" \
    --output-folder "${panlex_split_plain_folder}" \
    --source-suffix "plain.yid" --target-suffix "plain.deu" \
    --train-frac 0.70 --dev-frac 0.15 --test-frac 0.15

# Space separate and freq encode
for f in "${panlex_split_plain_folder}"/*
do
    fname=$(basename "${f}")
    sed "s/./& /g" < "${f}" | sed "s/\s$//" > "${panlex_split_space_separated_folder}/${fname//plain/space}"
    python order_stat_transform.py -s " " < "${f}" > "${panlex_split_freq_encoded_folder}/${fname//plain/freq}"
done

# Create freq -> space sep folder
mkdir -p $panlex_split_freq2space_folder
for f in "${panlex_split_space_separated_folder}"/*.deu 
do
    fname=$(basename "${f}")
    cp "${f}" "$panlex_split_freq2space_folder/${fname//.space/}"
done

for f in "${panlex_split_freq_encoded_folder}"/*.yid
do
    fname=$(basename "${f}")
    cp "${f}" "$panlex_split_freq2space_folder/${fname//.freq/}"
done
