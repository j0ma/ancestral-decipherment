from collections import Counter

import helpers as h
import click


@click.command()
@click.option("-s", "--separator", default=" ")
@click.option("--split-incoming-line", is_flag=True)
@click.option("--include-key", is_flag=True, help="Include 1:1 encryption key in output")
def main(separator, split_incoming_line, include_key):
    for line in click.get_text_stream("stdin"):
        chars = line.split(separator) if split_incoming_line else line
        freqs = {k: str(v) for k, v in h.gen_order_stat_map(chars).items()}
        encrypted = h.encrypt(chars, freqs, sep=separator)
        click.echo(f"{encrypted}\t{freqs}" if include_key else encrypted, nl=True)


if __name__ == "__main__":
    main()
