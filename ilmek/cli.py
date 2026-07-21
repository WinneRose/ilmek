"""Command-line interface: ``ilmek <command> ...``.

Implemented in v0.1: ``analyze``, ``lemmatize``, ``stem``. Reserved subcommands
(``models``, ``benchmark``, ``serve``) print an honest "arrives in vX" message rather than
pretending to work — no silent no-ops.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__, analyze_sentence
from .morphology.analyzer import default_analyzer


def _cmd_analyze(args: argparse.Namespace) -> int:
    text = " ".join(args.text)
    doc = analyze_sentence(text)
    if args.json:
        payload = [a.to_dict() if a is not None else None for a in doc.analyses]
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0
    for token, analysis in zip(doc.tokens, doc.analyses, strict=True):
        if analysis is None:
            continue
        morphemes = "+".join([analysis.lemma, *analysis.morphemes])
        feats = analysis.feats_string()
        print(
            f"{token.text}\t{analysis.lemma}\t{analysis.pos}\t{morphemes}\t{feats}"
            f"\t[{analysis.source}]"
        )
    return 0


def _cmd_lemmatize(args: argparse.Namespace) -> int:
    doc = analyze_sentence(" ".join(args.text))
    lemmas = [a.lemma for a in doc.analyses if a is not None]
    print(" ".join(lemmas))
    return 0


def _cmd_stem(args: argparse.Namespace) -> int:
    doc = analyze_sentence(" ".join(args.text))
    stems = [a.stem for a in doc.analyses if a is not None]
    print(" ".join(stems))
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    if args.models_command == "list":
        print("Native backend: built-in (no download required).")
        print("Optional model backends (stanza, zemberek): not installed.")
        return 0
    print("Model download arrives in v0.2 (Stanza adapter + model registry).", file=sys.stderr)
    return 1


def _cmd_benchmark(_args: argparse.Namespace) -> int:
    print("The evaluation/benchmark suite arrives in v0.4.", file=sys.stderr)
    return 1


def _cmd_serve(_args: argparse.Namespace) -> int:
    print(
        "The local HTTP API arrives in v0.5 (install extras: ilmek[server]).",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ilmek",
        description="Explainable Turkish morphology: stemmer, lemmatizer, analyzer.",
    )
    parser.add_argument("--version", action="version", version=f"ilmek {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Morphologically analyze text.")
    p_analyze.add_argument("text", nargs="+", help="Text to analyze.")
    p_analyze.add_argument("--json", action="store_true", help="Emit JSON.")
    p_analyze.set_defaults(func=_cmd_analyze)

    p_lemma = sub.add_parser("lemmatize", help="Print the lemma of each word.")
    p_lemma.add_argument("text", nargs="+")
    p_lemma.set_defaults(func=_cmd_lemmatize)

    p_stem = sub.add_parser("stem", help="Print the stem of each word.")
    p_stem.add_argument("text", nargs="+")
    p_stem.set_defaults(func=_cmd_stem)

    p_models = sub.add_parser("models", help="Manage optional model backends.")
    p_models.add_argument("models_command", choices=["list", "download"], nargs="?", default="list")
    p_models.add_argument("name", nargs="?", help="Model name (for download).")
    p_models.set_defaults(func=_cmd_models)

    p_bench = sub.add_parser("benchmark", help="Run the evaluation suite (v0.4).")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_serve = sub.add_parser("serve", help="Run the local HTTP API (v0.5).")
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Warm the analyzer once so first-word latency is not attributed to a subcommand.
    if args.command in {"analyze", "lemmatize", "stem"}:
        default_analyzer()
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
