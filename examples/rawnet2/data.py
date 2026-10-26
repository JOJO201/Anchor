"""Manifest-based audio loading for the RawNet2 example."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch.utils.data import Dataset


LABELS = {"spoof": 0, "bonafide": 1, "0": 0, "1": 1, 0: 0, 1: 1}


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {source}:{line_number}")
            missing = {"sample_id", "audio_path", "label"}.difference(row)
            if missing:
                raise ValueError(
                    f"missing {sorted(missing)} at {source}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {source}")
    return rows


def load_waveform(
    path: str | Path,
    *,
    sample_rate: int,
    num_samples: int,
    random_crop: bool,
) -> torch.Tensor:
    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        divisor = math.gcd(int(source_rate), int(sample_rate))
        audio = resample_poly(
            audio, sample_rate // divisor, source_rate // divisor
        ).astype(np.float32, copy=False)
    if audio.size == 0:
        raise ValueError(f"empty audio file: {path}")
    if audio.size < num_samples:
        repeats = int(math.ceil(num_samples / audio.size))
        audio = np.tile(audio, repeats)
    if audio.size > num_samples:
        maximum = audio.size - num_samples
        start = int(np.random.randint(0, maximum + 1)) if random_crop else 0
        audio = audio[start : start + num_samples]
    return torch.from_numpy(np.asarray(audio[:num_samples], dtype=np.float32))


class AudioManifestDataset(Dataset[dict[str, Any]]):
    """JSONL dataset with ``sample_id``, ``audio_path``, and ``label`` fields."""

    def __init__(
        self,
        manifest: str | Path,
        audio_root: str | Path,
        *,
        sample_rate: int = 16_000,
        num_samples: int = 64_600,
        random_crop: bool = False,
    ) -> None:
        self.rows = _read_manifest(manifest)
        self.audio_root = Path(audio_root).expanduser().resolve()
        self.sample_rate = int(sample_rate)
        self.num_samples = int(num_samples)
        self.random_crop = bool(random_crop)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        raw_label = row["label"]
        key = raw_label.lower() if isinstance(raw_label, str) else raw_label
        if key not in LABELS:
            raise ValueError(f"unsupported label {raw_label!r}")
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = self.audio_root / audio_path
        return {
            "sample_id": str(row["sample_id"]),
            "waveform": load_waveform(
                audio_path,
                sample_rate=self.sample_rate,
                num_samples=self.num_samples,
                random_crop=self.random_crop,
            ),
            "label": int(LABELS[key]),
        }
