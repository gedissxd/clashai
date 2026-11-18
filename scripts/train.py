"""Train the AI on your recordings (see docs/ai-training.md).

    uv sync --extra cpu --extra train
    uv run python scripts/train.py

The model is saved to ai_models/policy.onnx (plus policy.json beside it).
Turn it on with ai.enabled in config.yaml.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from clashroyalebuildabot.ai.dataset import build_examples, find_recordings
from clashroyalebuildabot.ai.features import (
    FEATURE_VERSION,
    GRID_CHANNELS,
    GRID_SHAPE,
    VEC_HP,
    VEC_SIZE,
    WAIT,
)
from clashroyalebuildabot.ai.policy import meta_path_for
from clashroyalebuildabot.constants import APP_DIR

DEFAULT_RECORDINGS = os.path.join(APP_DIR, "recordings")
DEFAULT_OUT = os.path.join(APP_DIR, "ai_models", "policy.onnx")
NEG = -1e9


def split_groups(groups, val_fraction, seed):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = int(round(len(unique) * val_fraction))
    if len(unique) >= 2:
        n_val = max(n_val, 1)
    val_groups = set(unique[:n_val].tolist())
    is_val = np.array([g in val_groups for g in groups])
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def to_tensors(ex, idx, device, flip=False):
    import torch

    grid = torch.from_numpy(ex.grid[idx].astype(np.float32)).to(device)
    vec = torch.from_numpy(ex.vec[idx]).to(device)
    card_mask = torch.from_numpy(ex.card_mask[idx]).to(device)
    tile_masks = torch.from_numpy(ex.tile_masks[idx]).to(device)
    card = torch.from_numpy(ex.card[idx]).to(device)
    tile_x = torch.from_numpy(ex.tile_x[idx]).to(device)
    tile_y = torch.from_numpy(ex.tile_y[idx]).to(device)

    if flip:
        # Mirror half the batch left-to-right (see features.flip_horizontal).
        f = torch.rand(len(idx), device=device) < 0.5
        grid[f] = grid[f].flip(-1)
        tile_masks[f] = tile_masks[f].flip(-1)
        hp = vec[f, VEC_HP.start : VEC_HP.stop]
        vec[f, VEC_HP.start : VEC_HP.stop] = hp[:, [1, 0, 3, 2]]
        playing = f & (card != WAIT)
        tile_x[playing] = GRID_SHAPE[1] - 1 - tile_x[playing]
    return grid, vec, card_mask, tile_masks, card, tile_x, tile_y


def losses(model, batch, wait_weight):
    import torch
    import torch.nn.functional as F

    grid, vec, card_mask, tile_masks, card, tile_x, tile_y = batch
    card_logits, tile_logits = model(grid, vec)
    card_logits = card_logits.masked_fill(~card_mask, NEG)
    weights = torch.where(card == WAIT, wait_weight, 1.0)
    card_loss = F.cross_entropy(card_logits, card, reduction="none")
    card_loss = (card_loss * weights).sum() / weights.sum()

    playing = card != WAIT
    if playing.any():
        rows = torch.nonzero(playing).squeeze(1)
        logits = tile_logits[rows, card[rows]]
        mask = tile_masks[rows, card[rows]]
        logits = logits.masked_fill(~mask, NEG).flatten(1)
        target = tile_y[rows] * GRID_SHAPE[1] + tile_x[rows]
        tile_loss = F.cross_entropy(logits, target)
    else:
        tile_loss = card_logits.new_zeros(())
    return card_logits, tile_logits, card_loss, tile_loss


def evaluate(model, ex, idx, device, batch_size, wait_weight):
    import torch

    model.eval()
    totals = dict(loss=0.0, n=0, play_n=0, card_ok=0, tile_ok=0, decide_ok=0)
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            b_idx = idx[start : start + batch_size]
            batch = to_tensors(ex, b_idx, device)
            card_logits, tile_logits, card_loss, tile_loss = losses(
                model, batch, wait_weight
            )
            _, _, card_mask, tile_masks, card, tile_x, tile_y = batch
            n = len(b_idx)
            totals["loss"] += (card_loss + tile_loss).item() * n
            totals["n"] += n

            # Did it make the same play/wait choice as you?
            pred = card_logits.argmax(1)
            totals["decide_ok"] += (
                ((pred == WAIT) == (card == WAIT)).sum().item()
            )

            playing = card != WAIT
            if playing.any():
                rows = torch.nonzero(playing).squeeze(1)
                no_wait = card_logits[rows].clone()
                no_wait[:, WAIT] = NEG
                totals["card_ok"] += (
                    (no_wait.argmax(1) == card[rows]).sum().item()
                )
                logits = tile_logits[rows, card[rows]].masked_fill(
                    ~tile_masks[rows, card[rows]], NEG
                )
                flat = logits.flatten(1).argmax(1)
                py, px = flat // GRID_SHAPE[1], flat % GRID_SHAPE[1]
                close = ((py - tile_y[rows]).abs() <= 1) & (
                    (px - tile_x[rows]).abs() <= 1
                )
                totals["tile_ok"] += close.sum().item()
                totals["play_n"] += len(rows)
    n, play_n = max(totals["n"], 1), max(totals["play_n"], 1)
    return {
        "loss": totals["loss"] / n,
        "play_or_wait_accuracy": totals["decide_ok"] / n,
        "card_accuracy": totals["card_ok"] / play_n,
        "tile_within_1_accuracy": totals["tile_ok"] / play_n,
    }


def export(model, out_path, deck, info):
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    model = model.cpu().eval()
    grid = torch.zeros(1, GRID_CHANNELS, *GRID_SHAPE)
    vec = torch.zeros(1, VEC_SIZE)
    kwargs = dict(
        input_names=["grid", "vec"],
        output_names=["card_logits", "tile_logits"],
        dynamic_axes={
            "grid": {0: "batch"},
            "vec": {0: "batch"},
            "card_logits": {0: "batch"},
            "tile_logits": {0: "batch"},
        },
        opset_version=17,
    )
    try:
        torch.onnx.export(model, (grid, vec), out_path, dynamo=False, **kwargs)
    except TypeError:  # older PyTorch without the dynamo argument
        torch.onnx.export(model, (grid, vec), out_path, **kwargs)

    meta = {
        "feature_version": FEATURE_VERSION,
        "deck": [card.name for card in deck],
        "trained": datetime.now().isoformat(timespec="seconds"),
        **info,
    }
    with open(meta_path_for(out_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--recordings", default=DEFAULT_RECORDINGS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--wait-weight",
        type=float,
        default=1.0,
        help="Lower it if the trained bot waits too much, raise it if it "
        "spams cards.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu or cuda (default: auto)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit(
            "Training needs PyTorch. Install it with:\n"
            "  uv sync --extra cpu --extra train"
        )
    from clashroyalebuildabot.ai.model import PolicyNet

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dirs = find_recordings(args.recordings)
    if not dirs:
        sys.exit(f"No recordings in {args.recordings}. Record some first.")
    print(f"Reading {len(dirs)} recording(s)...")
    ex = build_examples(dirs)
    print("Deck:", ", ".join(card.name for card in ex.deck))
    for key, value in sorted(ex.stats.items()):
        print(f"  {key}: {value}")
    n_play = int((ex.card != WAIT).sum())
    if n_play < 50:
        print(
            f"\nOnly {n_play} play examples: the model will be weak. "
            "A few hundred battles is a good target."
        )

    train_idx, val_idx = split_groups(ex.group, args.val_fraction, args.seed)
    print(
        f"\n{len(train_idx)} training and {len(val_idx)} validation examples, "
        f"device={device}"
    )

    model = PolicyNet().to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, args.epochs
    )
    rng = np.random.default_rng(args.seed)

    best = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        order = rng.permutation(train_idx)
        total, count = 0.0, 0
        for b in range(0, len(order), args.batch_size):
            b_idx = order[b : b + args.batch_size]
            if len(b_idx) < 2:  # BatchNorm needs more than one example
                continue
            batch = to_tensors(ex, b_idx, device, flip=True)
            _, _, card_loss, tile_loss = losses(model, batch, args.wait_weight)
            loss = card_loss + tile_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(b_idx)
            count += len(b_idx)
        scheduler.step()

        line = f"epoch {epoch:3d}  train_loss {total / max(count, 1):.3f}"
        if len(val_idx):
            metrics = evaluate(
                model, ex, val_idx, device, args.batch_size, args.wait_weight
            )
            line += (
                f"  val_loss {metrics['loss']:.3f}"
                f"  play/wait {metrics['play_or_wait_accuracy']:.0%}"
                f"  card {metrics['card_accuracy']:.0%}"
                f"  tile±1 {metrics['tile_within_1_accuracy']:.0%}"
            )
            score = metrics["loss"]
        else:
            metrics = {}
            score = -epoch  # no validation data: keep the last epoch
        print(f"{line}  ({time.time() - start:.1f}s)")
        if best is None or score < best[0]:
            best = (score, epoch, metrics)
            best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    info = {
        "best_epoch": best[1],
        "validation": best[2],
        "examples": {"play": n_play, "wait": int(len(ex) - n_play)},
        "recordings": len(dirs),
    }
    export(model, args.out, ex.deck, info)
    print(f"\nSaved the model from epoch {best[1]} to {args.out}")
    print("Turn it on with 'ai: enabled: True' in config.yaml.")


if __name__ == "__main__":
    main()
