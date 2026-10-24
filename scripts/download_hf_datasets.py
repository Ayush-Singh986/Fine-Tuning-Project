import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASETS: dict[str, dict[str, str]] = {
    "ultrachat_200k": {
        "repo_id": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "license": "MIT",
        "purpose": "General multi-turn instruction/chat SFT; not curated JSON extraction labels.",
    },
    "nemotron_instruction_chat": {
        "repo_id": "nvidia/Nemotron-Instruction-Following-Chat-v1",
        "revision": "83dcd3aded0d289b0bbc018d3f9af4c5dd4005df",
        "license": "See upstream README: chat ODC-BY-1.0; structured_outputs CC-BY-4.0",
        "purpose": "General instruction and structured-output SFT; not invoice-specific labels.",
    },
    "gsm8k": {
        "repo_id": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "license": "MIT",
        "purpose": "Math reasoning benchmark/training data; not structured JSON extraction.",
    },
    "orca_dpo_pairs": {
        "repo_id": "argilla/distilabel-intel-orca-dpo-pairs",
        "revision": "0b10ec0df32c919f95126b203c8f5962b6875896",
        "license": "Apache-2.0",
        "purpose": "General chat preference pairs; not JSON-task-specific preferences.",
    },
    "opencodeinstruct": {
        "repo_id": "nvidia/OpenCodeInstruct",
        "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
        "license": "CC-BY-4.0",
        "purpose": "Python/code instruction SFT; not structured JSON extraction.",
    },
    "opcoder_opc_sft_stage1": {
        "repo_id": "OpenCoder-LLM/opc-sft-stage1",
        "revision": "1bcab575f5e2d1c1fd6652720418524c27b3d58b",
        "license": "MIT",
        "purpose": "Code SFT; not structured JSON extraction.",
    },
}
ALLOW_PATTERNS = ["README.md", "*.parquet", "*.jsonl", "*.json", "*.csv"]


def download_all(output_dir: Path, selected: list[str]) -> list[dict[str, Any]]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Install the downloader dependency with "
            "`python -m pip install -e '.[download]'`."
        ) from error

    summaries = []
    for name in selected:
        dataset = DATASETS[name]
        destination = output_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading {dataset['repo_id']} at {dataset['revision']} "
            f"to {destination}",
            flush=True,
        )
        snapshot_download(
            repo_id=dataset["repo_id"],
            repo_type="dataset",
            revision=dataset["revision"],
            local_dir=str(destination),
            allow_patterns=ALLOW_PATTERNS,
            max_workers=4,
        )
        files = sorted(
            path for path in destination.rglob("*") if path.is_file()
        )
        summary = {
            **dataset,
            "name": name,
            "path": str(destination),
            "files": [
                {"path": str(path.relative_to(destination)), "bytes": path.stat().st_size}
                for path in files
                if ".cache" not in path.parts
            ],
            "file_count": sum(".cache" not in path.parts for path in files),
            "bytes": sum(
                path.stat().st_size for path in files if ".cache" not in path.parts
            ),
        }
        summaries.append(summary)
        print(
            f"Finished {name}: {summary['file_count']} files, "
            f"{summary['bytes']:,} bytes",
            flush=True,
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the linked Hugging Face datasets at pinned revisions."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS],
        action="append",
        help="Dataset name; repeat to select multiple. Defaults to all six.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external"),
        help="Destination directory (default: data/external).",
    )
    args = parser.parse_args()
    requested = args.dataset or ["all"]
    if "all" in requested and len(requested) != 1:
        parser.error("'all' cannot be combined with individual dataset names.")
    selected = list(DATASETS) if requested == ["all"] else requested
    summaries = download_all(args.output_dir, selected)
    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "datasets": summaries,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
