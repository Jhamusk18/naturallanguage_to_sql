"""The ablation comparison must refuse mixed evidence and compute claims from counts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation.compare import main


def _write(
    tmp_path: Path, name: str, *, prompt: str, corr: int, halluc: int, ok: int,
    model: str = "m", incomplete: bool = False, fallback: int = 0,
) -> str:
    payload = {
        "config": {
            "provider": "groq", "model": model, "prompt_mode": prompt,
            "max_corrections": corr, "sample_rows": 3, "incomplete": incomplete,
            "llm_fallback_calls": fallback, "date": "2026-11-18T00:00:00+00:00",
        },
        "metrics": {
            "answerable": 50, "valid_sql_first_try": 0, "execution_success": ok,
            "execution_accuracy": ok, "hallucinated_first_try": halluc,
            "needed_correction": 0, "corrected_ok": 0,
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_reduction_is_computed_from_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write(tmp_path, "b0.json", prompt="baseline", corr=0, halluc=40, ok=5)
    b = _write(tmp_path, "a0.json", prompt="schema_aware", corr=0, halluc=10, ok=40)
    c = _write(tmp_path, "a2.json", prompt="schema_aware", corr=2, halluc=10, ok=45)
    assert main([a, b, c]) == 0
    out = capsys.readouterr().out
    assert "40/50 (80.0%) -> 10/50 (20.0%); relative reduction 75.0%" in out
    assert "across two runs (schema_aware): execution success 40/50 (80.0%) -> 45/50" in out


def test_paired_correction_claim_from_one_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write(tmp_path, "b2.json", prompt="baseline", corr=2, halluc=40, ok=30)
    b = _write(tmp_path, "a2.json", prompt="schema_aware", corr=2, halluc=10, ok=45)
    for path, first in ((a, 20), (b, 35)):
        data = json.loads(Path(path).read_text())
        data["metrics"].update(execution_success_first_try=first, execution_accuracy_first_try=0)
        Path(path).write_text(json.dumps(data))
    assert main([a, b]) == 0
    out = capsys.readouterr().out
    assert "paired within one run (schema_aware" in out
    assert "query success 35/50 (70.0%) OFF -> 45/50 (90.0%) ON" in out


def test_refuses_different_models(tmp_path: Path) -> None:
    a = _write(tmp_path, "b0.json", prompt="baseline", corr=0, halluc=40, ok=5, model="x")
    b = _write(tmp_path, "a0.json", prompt="schema_aware", corr=0, halluc=10, ok=40, model="y")
    assert main([a, b]) == 1


@pytest.mark.parametrize("kwargs", [{"incomplete": True}, {"fallback": 3}])
def test_refuses_incomplete_or_mixed_provider_runs(tmp_path: Path, kwargs: dict) -> None:
    a = _write(tmp_path, "b0.json", prompt="baseline", corr=0, halluc=40, ok=5)
    b = _write(tmp_path, "a0.json", prompt="schema_aware", corr=0, halluc=10, ok=40, **kwargs)
    assert main([a, b]) == 1


def test_column_tolerant_match_allows_extra_columns_not_wrong_values() -> None:
    from evaluation.rescore import column_tolerant_match

    gold = [["Ann", 5], ["Bob", 3]]
    assert column_tolerant_match([[1, "Ann", "a@x", 5], [2, "Bob", "b@x", 3]], gold)
    assert not column_tolerant_match([["Ann", 5], ["Bob", 4]], gold)  # wrong value
    assert not column_tolerant_match([["Ann", 5]], gold)  # missing row
