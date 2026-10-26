"""Evaluate a RawNet2 + AnchorADD checkpoint and export paper-facing metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from anchoradd import AnchorBank, compute_binary_metrics

from .data import AudioManifestDataset
from .model import build_anchoradd_rawnet2
from .utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported checkpoint")
    if payload.get("anchor_source_split") != "train":
        raise ValueError("checkpoint anchors are not marked as train-only")
    return payload


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    set_seed(int(config.get("seed", 1234)))
    audio = config["audio"]
    dataset = AudioManifestDataset(
        args.manifest,
        args.audio_root,
        sample_rate=int(audio["sample_rate"]),
        num_samples=int(audio["num_samples"]),
        random_crop=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = build_anchoradd_rawnet2(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    anchor_ids = tuple(str(value) for value in checkpoint["anchor_ids"])
    bank = AnchorBank(checkpoint["anchor_bank"].float().cpu(), anchor_ids)
    anchors = bank.fixed(
        int(config["anchoradd"]["eval_anchor_count"]), device=device
    )

    records: list[tuple[str, int, float, float]] = []
    model.eval()
    for batch in tqdm(loader, desc="evaluate"):
        waveforms = batch["waveform"].to(device, non_blocking=True)
        output = model.predict(waveforms, anchors)
        probabilities = output["positive_probability"].cpu().numpy()
        uncertainty = output["uncertainty"].cpu().numpy()
        labels = batch["label"].numpy()
        for sample_id, label, score, variance in zip(
            batch["sample_id"], labels, probabilities, uncertainty
        ):
            records.append(
                (str(sample_id), int(label), float(score), float(variance))
            )

    labels = np.asarray([row[1] for row in records], dtype=np.int64)
    scores = np.asarray([row[2] for row in records], dtype=np.float64)
    variances = np.asarray([row[3] for row in records], dtype=np.float64)
    metrics = compute_binary_metrics(
        labels,
        scores,
        uncertainty=variances,
        ece_bins=int(config.get("evaluation", {}).get("ece_bins", 10)),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "p_bonafide", "anchor_variance"])
        writer.writerows(records)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
