import argparse
import json
from pathlib import Path
from typing import Any

METRICS = (
    ("JSON parse rate", "json_parse_rate"),
    ("Exact-match accuracy", "exact_match_accuracy"),
    ("Field-value accuracy", "field_value_accuracy"),
)


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        report = json.load(source)
    for label, key in METRICS:
        if not isinstance(report.get(key), (int, float)):
            raise ValueError(f"{path}: missing numeric metric {key!r} ({label}).")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base, SFT, and DPO evaluation reports.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--dpo", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional Markdown report path.")
    args = parser.parse_args()

    reports = {
        "Base prompted": load_report(args.base),
        "After SFT": load_report(args.sft),
        "After DPO": load_report(args.dpo),
    }
    counts = {report.get("examples") for report in reports.values()}
    if len(counts) != 1:
        raise ValueError("All reports must use the same number of held-out examples.")
    test_hashes = {report.get("test_set_sha256") for report in reports.values()}
    if len(test_hashes) != 1 or None in test_hashes:
        raise ValueError("All reports must use the same held-out test-set contents.")
    generations = {
        json.dumps(report.get("generation"), sort_keys=True)
        for report in reports.values()
    }
    if len(generations) != 1 or "null" in generations:
        raise ValueError("All reports must use the same decoding settings.")

    lines = [
        "| Metric | Base prompted | After SFT | After DPO |",
        "|---|---:|---:|---:|",
    ]
    for label, key in METRICS:
        values = [f"{reports[stage][key] * 100:.2f}%" for stage in reports]
        lines.append(f"| {label} | {' | '.join(values)} |")
    schema_reports = [
        report.get("schema_valid_rate")
        for report in reports.values()
    ]
    if all(isinstance(value, (int, float)) for value in schema_reports):
        values = [f"{value * 100:.2f}%" for value in schema_reports]
        lines.append(f"| Schema-valid rate | {' | '.join(values)} |")
    elif any(value is not None for value in schema_reports):
        raise ValueError("Schema-valid rate must be available in all reports or none.")
    else:
        lines.append("| Schema-valid rate | N/A | N/A | N/A |")
    lines.append(f"\nHeld-out examples: {next(iter(counts))}")
    result = "\n".join(lines)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
