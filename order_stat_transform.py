from collections import Counter

import helpers as h
import click


@click.command()
@click.option("-s", "--separator", default=" ")
@click.option("--include-key", is_flag=True, help="Include 1:1 encryption key in output")
def main(separator: str, include_key: bool) -> None:
    for line in click.get_text_stream("stdin"):
        freqs = {k: str(v) for k, v in h.gen_order_stat_map(line).items()}
        encrypted = h.encrypt(line, freqs, sep=separator)
        click.echo(f"{encrypted}\t{freqs}" if include_key else encrypted, nl=True)


if __name__ == "__main__":
    main()
