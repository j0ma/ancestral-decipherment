from collections import Counter

import helpers as h
import click


@click.command()
@click.option("-s", "--separator", default=" ")
def main(separator):
    for line in click.get_text_stream("stdin"):
        freqs = {k: str(v) for k, v in h.gen_order_stat_map(line).items()}
        click.echo(h.encrypt(line, freqs, sep=separator))


if __name__ == "__main__":
    main()
