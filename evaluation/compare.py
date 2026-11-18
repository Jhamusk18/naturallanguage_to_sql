"""Compare ablation result files produced by ``python -m evaluation.run --out ...``.

    python -m evaluation.compare A.json B.json [C.json ...]
    python -m evaluation.compare evaluation/results/groq/*.json

Prints one row per file, then every claim the files can support, computed
from the stored counts (never typed by hand):

* hallucination reduction: for each weaker prompt P with corrections off,
  (halluc(P) - halluc(schema_aware)) / halluc(P), first attempt only;
* self-correction lift: for each prompt, execution success with
  max_corrections=0 vs the same prompt with corrections on.

Refuses to compare files from different providers/models, incomplete runs,
or runs where any call was served by a fallback model: those cannot back a
single-model number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

WEAKER_PROMPTS = ("baseline", "columns_only")


def _rate(metrics: dict[str, Any], key: str) -> float:
    total = metrics["answerable"]
    return metrics[key] / total if total else 0.0


def _frac(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.1f}%)" if den else f"{num}/0"


def _problems(runs: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    problems: list[str] = []
    models = {(r["config"]["provider"], r["config"]["model"]) for _, r in runs}
    if len(models) > 1:
        problems.append(f"files come from different provider/model pairs: {sorted(models)}")
    for path, run in runs:
        config = run["config"]
        if config.get("incomplete"):
            problems.append(f"{path.name}: run is incomplete (provider error mid-run)")
        if config.get("llm_fallback_calls"):
            problems.append(
                f"{path.name}: {config['llm_fallback_calls']} calls served by a fallback model"
            )
    return problems


def _key(run: dict[str, Any]) -> tuple[str, int, int]:
    c = run["config"]
    prompt = c["prompt_mode"]
    if c.get("generation_mode", "text") == "tools":
        prompt += "+tools"
    return prompt, int(c["max_corrections"]), int(c.get("sample_rows", 3))


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    force = "--force" in args
    paths = [Path(a) for a in args if a != "--force"]
    if len(paths) < 2:
        print(__doc__)
        return 2
    runs = [(p, json.loads(p.read_text())) for p in paths]

    problems = _problems(runs)
    for problem in problems:
        print(f"!! {problem}")
    if problems and not force:
        print("Refusing to compare (pass --force to print the table anyway; claims are skipped).")
        return 1

    provider, model = runs[0][1]["config"]["provider"], runs[0][1]["config"]["model"]
    dates = sorted(r["config"]["date"] for _, r in runs)
    print(f"provider={provider} model={model} runs {dates[0]} .. {dates[-1]}\n")
    print(
        f"{'file':<24}{'prompt':<14}{'samp':>5}{'corr':>5}{'n':>4}"
        f"{'halluc':>9}{'valid1':>9}{'exec':>9}{'acc':>9}{'fixed':>9}"
    )
    for path, run in runs:
        m, c = run["metrics"], run["config"]
        fixed = f"{m['corrected_ok']}/{m['needed_correction']}"
        print(
            f"{path.name:<24}{c['prompt_mode']:<14}{c.get('sample_rows', '-'):>5}"
            f"{c['max_corrections']:>5}{m['answerable']:>4}"
            f"{_rate(m, 'hallucinated_first_try'):>9.1%}{_rate(m, 'valid_sql_first_try'):>9.1%}"
            f"{_rate(m, 'execution_success'):>9.1%}{_rate(m, 'execution_accuracy'):>9.1%}"
            f"{fixed:>9}"
        )
    if problems:
        return 1

    by_key = {_key(run): run["metrics"] for _, run in runs}
    print("\nClaims supported by these files:")
    for corr in sorted({k for _p, k, _s in by_key}):
        aware = by_key.get(("schema_aware", corr, 3))
        if aware is None:
            continue
        for weaker in WEAKER_PROMPTS:
            base = by_key.get((weaker, corr, 3))
            if base is None:
                continue
            hb, ha = base["hallucinated_first_try"], aware["hallucinated_first_try"]
            if hb:
                print(
                    f"- hallucinated first-try SQL, {weaker} -> schema_aware (max {corr} "
                    "retries): "
                    f"{_frac(hb, base['answerable'])} -> {_frac(ha, aware['answerable'])}; "
                    f"relative reduction {(hb - ha) / hb:.1%}"
                )
        no_samples = by_key.get(("schema_aware", corr, 0))
        if no_samples is not None:
            print(
                "- sample rows (schema_aware), 0 -> 3 rows: hallucinated "
                f"{_frac(no_samples['hallucinated_first_try'], no_samples['answerable'])} -> "
                f"{_frac(aware['hallucinated_first_try'], aware['answerable'])}; accuracy "
                f"{_frac(no_samples['execution_accuracy'], no_samples['answerable'])} -> "
                f"{_frac(aware['execution_accuracy'], aware['answerable'])}"
            )
    for (prompt, corr, samples), m in sorted(by_key.items()):
        if corr == 0 or "execution_success_first_try" not in m:
            continue
        n = m["answerable"]
        print(
            f"- self-correction, paired within one run ({prompt}, sample_rows={samples}, "
            f"max {corr} retries): query success "
            f"{_frac(m['execution_success_first_try'], n)} OFF -> "
            f"{_frac(m['execution_success'], n)} ON; accuracy "
            f"{_frac(m['execution_accuracy_first_try'], n)} -> "
            f"{_frac(m['execution_accuracy'], n)}; "
            f"{m['corrected_ok']}/{m['needed_correction']} failed first attempts fixed"
        )
    for prompt in ("schema_aware", *WEAKER_PROMPTS):
        off = by_key.get((prompt, 0, 3))
        on = next(
            (m for (p, k, s), m in by_key.items() if p == prompt and k > 0 and s == 3), None
        )
        if off is None or on is None:
            continue
        print(
            f"- self-correction across two runs ({prompt}): execution success "
            f"{_frac(off['execution_success'], off['answerable'])} -> "
            f"{_frac(on['execution_success'], on['answerable'])}; accuracy "
            f"{_frac(off['execution_accuracy'], off['answerable'])} -> "
            f"{_frac(on['execution_accuracy'], on['answerable'])}"
        )
    for path, run in runs:
        m = run["metrics"]
        if "nonexistent_total" in m:
            print(
                f"- {path.name}: unsafe rejected {m['unsafe_rejected']}/{m['unsafe_total']}, "
                f"nonexistent not executed {m['nonexistent_not_executed']}/"
                f"{m['nonexistent_total']}, ambiguous not executed "
                f"{m['ambiguous_not_executed']}/{m['ambiguous_total']}, answerable falsely "
                f"declined {m['answerable_declined']}/{m['answerable']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
