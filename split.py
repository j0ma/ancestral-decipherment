#!/usr/bin/env python

from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
import click


def files_have_equal_nr_lines(source_file, target_file):
    file_sizes = Counter()

    for f in [source_file, target_file]:
        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                file_sizes[f] += 1

    assert (
        len(set(file_sizes.values())) == 1
    ), f"All files should have the same number of lines. Got: {file_sizes}"


def fractions_sum_to_one(train_frac, dev_frac, test_frac):
    frac_sum = train_frac + dev_frac + test_frac
    assert (
        frac_sum == 1.0
    ), f"Incompatible train/dev/test fractions! They sum to {frac_sum} != 1.0"


@click.command()
@click.option("--source-file", type=click.Path(readable=True, path_type=Path))
@click.option("--target-file", type=click.Path(readable=True, path_type=Path))
@click.option(
    "--output-folder", type=click.Path(dir_okay=True, file_okay=False, path_type=Path)
)
@click.option("--train-frac", type=float)
@click.option("--dev-frac", type=float)
@click.option("--test-frac", type=float)
@click.option("--source-suffix", default="freq")
@click.option("--target-suffix", default="space")
def main(
    source_file,
    target_file,
    output_folder,
    train_frac,
    dev_frac,
    test_frac,
    source_suffix,
    target_suffix,
):

    # Check conditions
    fractions_sum_to_one(train_frac, dev_frac, test_frac)
    files_have_equal_nr_lines(source_file, target_file)

    # Generate pairs and split
    with open(source_file, encoding="utf-8") as src_in, open(
        target_file, encoding="utf-8"
    ) as tgt_in:
        pairs = [(src, tgt) for src, tgt in zip(src_in, tgt_in)]
        pairs_train, pairs_rest = train_test_split(
            pairs, train_size=train_frac, test_size=(dev_frac + test_frac)
        )
        pairs_dev, pairs_test = train_test_split(
            pairs_rest,
            train_size=dev_frac / (dev_frac + test_frac),
            test_size=test_frac / (dev_frac + test_frac),
        )

    # Write train

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    for split_name, split_pairs in zip(
        ["train", "dev", "test"], [pairs_train, pairs_dev, pairs_test]
    ):
        src_out = output_folder / f"{split_name}.{source_suffix}"
        tgt_out = output_folder / f"{split_name}.{target_suffix}"
        with open(src_out, encoding="utf-8", mode="w") as f_src_out, open(
            tgt_out, encoding="utf-8", mode="w"
        ) as f_tgt_out:
            for (_src, _tgt) in split_pairs:
                click.echo(message=_src, file=f_src_out, nl=False)
                click.echo(message=_tgt, file=f_tgt_out, nl=False)


if __name__ == "__main__":
    main()
