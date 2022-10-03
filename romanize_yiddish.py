#!/usr/bin/env python

import click
import yiddish


def transliterate(s: str) -> str:
    return yiddish.transliterate(s, loshn_koydesh=True)


@click.command()
def main():
    for line in click.get_text_stream("stdin"):
        click.echo(transliterate(line), nl=False)


if __name__ == "__main__":
    main()
