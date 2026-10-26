#!/usr/bin/env python3
"""Convert an ASVspoof-style protocol into the example JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--audio-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--id-column", type=int, default=1)
    parser.add_argument("--label-column", type=int, default=-1)
    parser.add_argument("--extension", default=".flac")
    parser.add_argument("--audio-root", help="if supplied, verify every audio file")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = Path(args.protocol).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    audio_root = (
        None if args.audio_root is None else Path(args.audio_root).expanduser().resolve()
    )
    rows = []
    with protocol.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            columns = line.split()
            if not columns:
                continue
            try:
                sample_id = columns[args.id_column]
                label = columns[args.label_column].lower()
            except IndexError as exc:
                raise ValueError(f"not enough columns at {protocol}:{line_number}") from exc
            if label not in {"spoof", "bonafide"}:
                raise ValueError(f"unsupported label {label!r} at {protocol}:{line_number}")
            relative = Path(args.audio_prefix) / f"{sample_id}{args.extension}"
            if audio_root is not None and not (audio_root / relative).is_file():
                raise FileNotFoundError(audio_root / relative)
            rows.append(
                {
                    "sample_id": sample_id,
                    "audio_path": relative.as_posix(),
                    "label": label,
                    "split": args.split,
                }
            )
    if not rows:
        raise ValueError(f"protocol contains no usable rows: {protocol}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "records": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
