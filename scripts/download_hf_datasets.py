import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET = {
    "repo_id": "nvidia/Nemotron-Instruction-Following-Chat-v1",
    "revision": "83dcd3aded0d289b0bbc018d3f9af4c5dd4005df",
    "license": "See upstream README: chat ODC-BY-1.0; structured_outputs CC-BY-4.0",
    "purpose": "JSON-formatted structured-output SFT examples.",
}
ALLOW_PATTERNS = ["README.md", "data/*.jsonl"]


def download_dataset(output_dir: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Install the downloader dependency with "
            "`python -m pip install -e '.[download]'`."
        ) from error

    name = "nemotron_instruction_chat"
    destination = output_dir / name
    destination.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {DATASET['repo_id']} at {DATASET['revision']} "
        f"to {destination}",
        flush=True,
    )
    snapshot_download(
        repo_id=DATASET["repo_id"],
        repo_type="dataset",
        revision=DATASET["revision"],
        local_dir=str(destination),
        allow_patterns=ALLOW_PATTERNS,
        max_workers=4,
    )
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    summary = {
        **DATASET,
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
    print(
        f"Finished {name}: {summary['file_count']} files, "
        f"{summary['bytes']:,} bytes",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Nemotron dataset used for JSON SFT."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/external"),
        help="Destination directory (default: data/external).",
    )
    args = parser.parse_args()
    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [download_dataset(args.output_dir)],
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
