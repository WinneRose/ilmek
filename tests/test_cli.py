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
    assert main(["benchmark"]) == 1
    assert main(["serve"]) == 1
    err = capsys.readouterr().err
    assert "v0.4" in err or "v0.5" in err
