"""CLI smoke tests: analyze / lemmatize / stem via ``main``."""

from __future__ import annotations

import json

import pytest

from ilmek.cli import main


@pytest.mark.positive
def test_cli_lemmatize(capsys):
    rc = main(["lemmatize", "kitaplarımızdan"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out == "kitap"


@pytest.mark.positive
def test_cli_stem_sentence(capsys):
    rc = main(["stem", "Kitaplarımızı masaya bıraktık."])
    out = capsys.readouterr().out.strip().split()
    assert rc == 0
    assert out[:3] == ["kitap", "masa", "bırak"]


@pytest.mark.positive
def test_cli_analyze_json(capsys):
    rc = main(["analyze", "evi", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    lemmas = {entry["lemma"] for entry in payload if entry}
    assert "ev" in lemmas


@pytest.mark.positive
def test_cli_analyze_table(capsys):
    rc = main(["analyze", "geliyorum"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "gel" in out
    assert "VERB" in out


@pytest.mark.negative
def test_cli_reserved_commands_report_honestly(capsys):
    # Premise change (called out in the PR): ``benchmark`` is no longer reserved — the
    # evaluation harness landed this milestone, so it now returns 0 (asserted below). Only
    # ``serve`` remains a reserved, honestly-reported stub.
    assert main(["serve"]) == 1
    err = capsys.readouterr().err
    assert "v0.5" in err


@pytest.mark.positive
def test_cli_benchmark_runs_and_reports(capsys):
    rc = main(["benchmark"])
    out = capsys.readouterr().out
    assert rc == 0
    # The readable report names the core metrics and per-category rows.
    assert "lemma" in out
    assert "cover" in out
    assert "words/sec" in out
    assert "simple_noun" in out and "ambiguity" in out
    assert "OVERALL" in out


@pytest.mark.positive
def test_cli_benchmark_json_parses_with_stable_keys(capsys):
    rc = main(["benchmark", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert {"overall", "categories", "n"} <= set(payload)
    assert payload["overall"]["lemma_accuracy"]["total"] > 0
    assert isinstance(payload["categories"], dict)


@pytest.mark.negative
def test_cli_benchmark_missing_gold_errors(capsys):
    rc = main(["benchmark", "--gold", "does-not-exist.json"])
    err = capsys.readouterr().err
    assert rc != 0  # no silent fallback
    assert "not found" in err or "error" in err.lower()
