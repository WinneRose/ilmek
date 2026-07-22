"""Command-line interface: ``ilmek <command> ...``.

Implemented: ``analyze``, ``lemmatize``, ``stem``, and ``benchmark`` (the evaluation harness
landed ahead of its v0.4 label — the roadmap item ships now, the version string is just a
plan marker). The remaining reserved subcommands (``models``, ``serve``) print an honest
"arrives in vX" message rather than pretending to work — no silent no-ops.
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


def _cmd_benchmark(args: argparse.Namespace) -> int:
    # Lazy import (mirrors the disambiguator pattern): keeps `ilmek analyze` startup cost
    # unchanged, since the evaluation layer pulls in the pipeline + gold dataset.
    from .evaluation.benchmark import GoldError, run_benchmark

    try:
        report = run_benchmark(path=args.gold, category=args.category)
    except GoldError as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(report.to_dict(), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(report.format_report())
    return 0


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

    p_bench = sub.add_parser(
        "benchmark",
        help="Run the evaluation suite over the gold set (PYTHONUTF8=1 for Turkish output).",
    )
    p_bench.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    p_bench.add_argument(
        "--category", help="Restrict the run to one gold category (e.g. voice, ambiguity)."
    )
    p_bench.add_argument("--gold", help="Path to an alternative gold JSON file.")
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
