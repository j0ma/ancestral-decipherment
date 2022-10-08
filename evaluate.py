from collections import defaultdict
from typing import List, TextIO, Dict, Tuple, Set, Optional
import editdistance
import sacrebleu
import numpy as np
import jiwer
import click
import attr
import sys

import pandas as pd
from tqdm import tqdm

from typing import *

"""
Computes SER evaluation metrics for deciphered words
"""

def read_df(
    input_file: str,
    io_format: str,
    typ: str = "frame",
    chunksize: Union[int, None] = None,
    column_names: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    if io_format in ["csv", "tsv"]:
        return pd.read_csv(
            input_file,
            encoding="utf-8",
            delimiter="\t" if io_format == "tsv" else ",",
            chunksize=chunksize,
            na_values=set(
                [
                    "#N/A",
                    "#N/A N/A",
                    "#NA",
                    "-1.#IND",
                    "-1.#QNAN",
                    "-NaN",
                    "1.#IND",
                    "1.#QNAN",
                    "<NA>",
                    "N/A",
                    "NA",
                    "NULL",
                    "NaN",
                    "n/a",
                    "null",
                ]
            ),
            keep_default_na=False,
            names=column_names,
            **kwargs,
        )
    elif io_format == "jsonl":
        return pd.read_json(
            input_file,
            "records",
            encoding="utf-8",
            typ=typ,
            lines=True,
            chunksize=chunksize,
            **kwargs,
        )
    elif io_format == "json":
        return pd.read_json(
            input_file, encoding="utf-8", typ=typ, chunksize=chunksize, **kwargs
        )


def read_text(path: str) -> TextIO:
    return open(path, encoding="utf-8")


@attr.s(kw_only=True)  # kw_only ensures we are explicit
class DeciphermentOutput:
    """Represents a single transliteration output, consisting of a
    source language and line, reference transliteration and a model hypothesis.
    """

    language: str = attr.ib()
    reference: str = attr.ib()
    hypothesis: str = attr.ib()
    source: str = attr.ib(default="")


@attr.s(kw_only=True)
class DeciphermentMetrics:
    """Score container for a collection of transliteration results.

    Contains
    - LER
    - Word Accuracy
    """

    symbol_error_rate: float = attr.ib(factory=float)
    word_acc: float = attr.ib(factory=float)
    rounding: int = attr.ib(default=5)
    language: str = attr.ib(default="")

    def __attrs_post_init__(self) -> None:
        self.symbol_error_rate = round(self.symbol_error_rate, self.rounding)
        self.word_acc = round(self.word_acc, self.rounding)

    def format(self) -> str:
        """Format like in old evaluate.py"""

        out = """Word Accuracy\t{word_acc:.4f}
SER\t{symbol_error_rate:.4f}\n\n""".format(
            word_acc=self.word_acc,
            symbol_error_rate=self.symbol_error_rate,
        )

        return out


@attr.s(kw_only=True)
class DeciphermentResults:
    system_outputs: List[DeciphermentOutput] = attr.ib(factory=list)
    metrics: DeciphermentMetrics = attr.ib(factory=DeciphermentMetrics)

    def __attrs_post_init__(self) -> None:
        self.metrics = self.compute_metrics()

    def compute_metrics(self) -> DeciphermentMetrics:
        unique_languages = set([o.language for o in self.system_outputs])

        if len(unique_languages) > 1:
            language = "global"
        else:
            language = list(unique_languages)[0]

        symbol_error_rate = self.symbol_error_rate(self.system_outputs)
        word_acc = 100 * self.word_accuracy(self.system_outputs)

        metrics = DeciphermentMetrics(
            symbol_error_rate=symbol_error_rate,
            word_acc=word_acc,
            language=language,
        )

        return metrics

    def symbol_error_rate(self, system_outputs: List[DeciphermentOutput]) -> float:

        # The names are strings of space-separated characters.
        # Thus, to get SER on the original string, we compute WER
        # on the space-separated tokens.
        SER = jiwer.wer(
            [o.reference for o in system_outputs],
            [o.hypothesis for o in system_outputs],
        )

        return SER

    def word_accuracy(self, system_outputs: List[DeciphermentOutput]) -> float:
        return np.mean([int(o.reference == o.hypothesis) for o in system_outputs])


@attr.s(kw_only=True)
class ExperimentResults:
    system_outputs: List[DeciphermentOutput] = attr.ib(factory=list)
    languages: Set[str] = attr.ib(factory=set)
    grouped: bool = attr.ib(default=True)
    metrics_dict: Dict[str, DeciphermentResults] = attr.ib(factory=dict)

    def __attrs_post_init__(self) -> None:
        self.metrics_dict = self.compute_metrics_dict()

    def compute_metrics_dict(self) -> Dict[str, DeciphermentResults]:

        metrics = {}

        # first compute global metrics
        metrics["global"] = DeciphermentResults(system_outputs=self.system_outputs)

        # then compute one for each lang

        for lang in tqdm(self.languages, total=len(self.languages)):
            filtered_outputs = [o for o in self.system_outputs if o.language == lang]
            metrics[lang] = DeciphermentResults(system_outputs=filtered_outputs)

        return metrics

    @classmethod
    def outputs_from_paths(
        cls,
        references_path: str,
        hypotheses_path: str,
        source_path: str,
        languages_path: str,
    ) -> Tuple[List[DeciphermentOutput], Set[str]]:
        with read_text(hypotheses_path) as hyp, read_text(
            references_path
        ) as ref, read_text(source_path) as src, read_text(languages_path) as langs:
            languages = set()
            system_outputs = []

            for hyp_line, ref_line, src_line, langs_line in zip(hyp, ref, src, langs):

                # grab hypothesis lines
                hypothesis = hyp_line.strip()
                reference = ref_line.strip()
                source = src_line.strip()
                language = langs_line.strip()
                languages.add(language)
                system_outputs.append(
                    DeciphermentOutput(
                        language=language,
                        reference=reference,
                        hypothesis=hypothesis,
                        source=source,
                    )
                )

            return system_outputs, languages

    @classmethod
    def outputs_from_combined_tsv(
        cls, combined_tsv_path: str
    ) -> Tuple[List[DeciphermentOutput], Set[str]]:

        combined_tsv = read_df(
            combined_tsv_path,
            io_format="tsv",
            column_names=["ref", "hyp", "src", "language"],
            quoting=3,
        ).astype(str)

        languages = set()
        system_outputs = []

        for hypothesis, reference, source, language in tqdm(
            zip(
                combined_tsv.hyp,
                combined_tsv.ref,
                combined_tsv.src,
                combined_tsv.language,
            ),
            total=combined_tsv.shape[0],
        ):

            # grab hypothesis lines
            languages.add(language)
            system_outputs.append(
                DeciphermentOutput(
                    language=language,
                    reference=reference,
                    hypothesis=hypothesis,
                    source=source,
                )
            )

        return system_outputs, languages

    @classmethod
    def from_paths(
        cls,
        references_path: str,
        hypotheses_path: str,
        source_path: str,
        languages_path: str,
        grouped: bool = True,
    ):
        system_outputs, languages = cls.outputs_from_paths(
            references_path=references_path,
            hypotheses_path=hypotheses_path,
            source_path=source_path,
            languages_path=languages_path,
        )

        return ExperimentResults(
            system_outputs=system_outputs, grouped=grouped, languages=languages
        )

    @classmethod
    def from_tsv(
        cls,
        tsv_path: str,
        grouped: bool = True,
    ):
        system_outputs, languages = cls.outputs_from_combined_tsv(tsv_path)

        return ExperimentResults(
            system_outputs=system_outputs, grouped=grouped, languages=languages
        )

    def as_data_frame(self):
        _languages = self.languages | set(["global"])

        rows = [attr.asdict(self.metrics_dict[lang].metrics) for lang in _languages]
        out = (
            pd.DataFrame(rows)
            .drop(columns=["rounding"])
            .rename(
                columns={
                    "symbol_error_rate": "SER",
                    "word_acc": "Accuracy",
                    "language": "Language",
                }
            )
            .round(3)
        )

        return out


@click.command()
@click.option("--references-path", "--gold-path", "--ref", "--gold", default="")
@click.option("--hypotheses-path", "--hyp", default="")
@click.option("--source-path", "--src", default="")
@click.option("--languages-path", "--langs", default="")
@click.option("--combined-tsv-path", "--tsv", default="")
@click.option("--score-output-path", "--score", default="/dev/stdout")
@click.option("--output-as-tsv", is_flag=True)
@click.option("--output-as-json", is_flag=True)
def main(
    references_path: str,
    hypotheses_path: str,
    source_path: str,
    languages_path: str,
    combined_tsv_path: str,
    score_output_path: str,
    output_as_tsv: bool,
    output_as_json: bool,
):

    if combined_tsv_path:
        results = ExperimentResults.from_tsv(tsv_path=combined_tsv_path)
    else:
        results = ExperimentResults.from_paths(
            references_path=references_path,
            hypotheses_path=hypotheses_path,
            source_path=source_path,
            languages_path=languages_path,
        )

    if output_as_tsv:
        result_df = results.as_data_frame()
        result_df.to_csv(score_output_path, index=False, sep="\t")
    else:
        with (
            open(score_output_path, "w", encoding="utf-8")

            if score_output_path
            else sys.stdout
        ) as score_out_file:
            for lang in results.languages:
                score_out_file.write(f"{lang}:\n")
                score_out_file.write(results.metrics_dict.get(lang).metrics.format())

            # finally write out global
            score_out_file.write("global:\n")
            score_out_file.write(results.metrics_dict.get("global").metrics.format())


if __name__ == "__main__":
    main()
