"""Train the RawNet2 + AnchorADD example from JSONL manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from anchoradd import stateless_anchor_index, stateless_apply_decision

from .data import AudioManifestDataset
from .model import build_anchoradd_rawnet2
from .utils import build_anchor_bank, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _make_loader(
    dataset: AudioManifestDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    anchors: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        waveforms = batch["waveform"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)
        output = model.predict(waveforms, anchors)
        probability = output["mean_probability"].clamp_min(1.0e-12)
        total_loss += float(F.nll_loss(probability.log(), labels, reduction="sum"))
        total_correct += int((probability.argmax(dim=1) == labels).sum())
        total += int(labels.numel())
    return total_loss / total, total_correct / total


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    anchor_waveforms: torch.Tensor,
    anchor_ids: tuple[str, ...],
    epoch: int,
    dev_loss: float,
) -> None:
    payload = {
        "schema_version": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "anchor_bank": anchor_waveforms.cpu(),
        "anchor_ids": anchor_ids,
        "anchor_source_split": "train",
        "epoch": int(epoch),
        "dev_loss": float(dev_loss),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 1234))
    set_seed(seed)
    device = torch.device(args.device)
    audio = config["audio"]
    training = config["training"]
    method = config["anchoradd"]

    train_set = AudioManifestDataset(
        args.train_manifest,
        args.audio_root,
        sample_rate=int(audio["sample_rate"]),
        num_samples=int(audio["num_samples"]),
        random_crop=bool(audio.get("random_crop_train", True)),
    )
    dev_set = AudioManifestDataset(
        args.dev_manifest,
        args.audio_root,
        sample_rate=int(audio["sample_rate"]),
        num_samples=int(audio["num_samples"]),
        random_crop=False,
    )
    anchor_bank = build_anchor_bank(
        train_set,
        bank_size=int(method["bank_size"]),
        seed=int(method.get("bank_seed", seed)),
    )
    eval_anchors = anchor_bank.fixed(
        int(method["eval_anchor_count"]), device=device
    )

    train_loader = _make_loader(
        train_set,
        batch_size=int(training["batch_size"]),
        num_workers=int(training.get("num_workers", 4)),
        shuffle=True,
        seed=seed,
    )
    dev_loader = _make_loader(
        dev_set,
        batch_size=int(training.get("eval_batch_size", training["batch_size"])),
        num_workers=int(training.get("num_workers", 4)),
        shuffle=False,
        seed=seed,
    )

    model = build_anchoradd_rawnet2(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
        amsgrad=bool(training.get("amsgrad", True)),
    )
    class_weights = torch.tensor(
        training.get("class_weights", [0.1, 0.9]),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    epochs = int(training["epochs"])
    delta = float(method["delta"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(progress):
            waveforms = batch["waveform"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            use_anchor = stateless_apply_decision(delta, seed, epoch, step)
            anchor = None
            if use_anchor:
                index = stateless_anchor_index(len(anchor_bank), seed, epoch, step)
                anchor = anchor_bank.one(index, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms, anchor)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = int(labels.numel())
            running_loss += float(loss.detach()) * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        dev_loss, dev_accuracy = validate(model, dev_loader, eval_anchors, device)
        print(
            f"epoch={epoch + 1} train_loss={running_loss / seen:.6f} "
            f"dev_loss={dev_loss:.6f} dev_accuracy={dev_accuracy:.4f}"
        )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            anchor_waveforms=anchor_bank.waveforms,
            anchor_ids=anchor_bank.sample_ids,
            epoch=epoch,
            dev_loss=dev_loss,
        )
        if dev_loss < best_loss:
            best_loss = dev_loss
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                anchor_waveforms=anchor_bank.waveforms,
                anchor_ids=anchor_bank.sample_ids,
                epoch=epoch,
                dev_loss=dev_loss,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
