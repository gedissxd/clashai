"""Play with a trained model (made by scripts/train.py)."""

import json
import os
from dataclasses import dataclass

import numpy as np

from clashroyalebuildabot.ai.features import (
    FEATURE_VERSION,
    WAIT,
    FeatureEncoder,
)
from clashroyalebuildabot.ai.serialize import card_by_name
from clashroyalebuildabot.namespaces import State


def meta_path_for(model_path: str) -> str:
    return os.path.splitext(model_path)[0] + ".json"


@dataclass(frozen=True)
class Decision:
    slot: int  # hand position 0-3
    tile_x: int
    tile_y: int
    card_name: str
    confidence: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def choose(
    card_logits: np.ndarray,
    tile_logits: np.ndarray,
    card_mask: np.ndarray,
    tile_masks: np.ndarray,
    playable: dict[int, int],
    play_threshold: float,
) -> tuple[int, int, int, float] | None:
    """Pick (deck index, tile_x, tile_y, confidence), or None to wait.

    The best playable card is played when, compared with waiting, the
    model is at least ``play_threshold`` sure about it.
    """
    probs = _softmax(np.where(card_mask, card_logits, -np.inf))
    candidates = [i for i in playable if tile_masks[i].any()]
    if not candidates:
        return None
    best = max(candidates, key=lambda i: probs[i])
    confidence = probs[best] / (probs[best] + probs[WAIT])
    if confidence < play_threshold:
        return None
    tiles = np.where(tile_masks[best], tile_logits[best], -np.inf)
    tile_y, tile_x = np.unravel_index(np.argmax(tiles), tiles.shape)
    return best, int(tile_x), int(tile_y), float(confidence)


class LearnedPolicy:
    def __init__(self, model_path: str, play_threshold: float = 0.5):
        import onnxruntime as ort

        meta_path = meta_path_for(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. "
                "Train one with scripts/train.py (see docs/ai-training.md)."
            )
        with open(meta_path, encoding="utf-8") as file:
            meta = json.load(file)
        if meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"{model_path} was trained with an older version of the "
                "bot. Train it again with scripts/train.py."
            )
        self.deck = [card_by_name(name) for name in meta["deck"]]
        self.encoder = FeatureEncoder(self.deck)
        self.play_threshold = play_threshold
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def decide(self, state: State) -> Decision | None:
        playable = self.encoder.playable(state)
        if not playable:
            return None
        grid, vec = self.encoder.encode(state)
        card_logits, tile_logits = self.session.run(
            ["card_logits", "tile_logits"],
            {"grid": grid[None], "vec": vec[None]},
        )
        choice = choose(
            card_logits[0],
            tile_logits[0],
            self.encoder.card_mask(state),
            self.encoder.tile_masks(state),
            playable,
            self.play_threshold,
        )
        if choice is None:
            return None
        index, tile_x, tile_y, confidence = choice
        return Decision(
            slot=playable[index],
            tile_x=tile_x,
            tile_y=tile_y,
            card_name=self.deck[index].name,
            confidence=confidence,
        )
