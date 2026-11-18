"""Turn recordings (see scripts/record.py) into training examples.

A recording folder holds:
    meta.json     deck, touch axis ranges and calibration
    states.jsonl  {"t": seconds, "state": {...}} for every screenshot
    touch.jsonl   {"t": seconds, "line": "<getevent -lt line>"}

Every in-game state becomes one example: either "play this card on this
tile" (the states just before you picked a card) or "wait".
"""

import bisect
import json
import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from clashroyalebuildabot.ai.features import (
    WAIT,
    FeatureEncoder,
    nearest_allowed_tile,
)
from clashroyalebuildabot.ai.serialize import card_by_name, state_from_dict
from clashroyalebuildabot.ai.touch import (
    Calibration,
    GestureBuilder,
    Play,
    gestures_to_plays,
    parse_event_line,
)
from clashroyalebuildabot.namespaces import Screens

# States this long before you pick a card are labelled with that play.
LEAD_TIME = 0.6
# After placing, the card must leave that hand position within this long,
# or the play is treated as a misclick and skipped.
CONFIRM_TIME = 3.0
# Each minute of play is kept together when splitting train/validation.
CHUNK_SECONDS = 60

RECORDING_VERSION = 1


@dataclass
class Recording:
    path: str
    deck: list
    times: list[float]
    states: list
    plays: list[Play]


@dataclass
class Examples:
    grid: np.ndarray  # uint8 (N, C, 32, 18)
    vec: np.ndarray  # float32 (N, V)
    card_mask: np.ndarray  # bool (N, 9)
    tile_masks: np.ndarray  # bool (N, 8, 32, 18)
    card: np.ndarray  # int64 (N,) deck index or WAIT
    tile_x: np.ndarray  # int64 (N,) -1 for WAIT
    tile_y: np.ndarray
    group: np.ndarray  # int64 (N,) for the train/validation split
    deck: list = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)

    def __len__(self):
        return len(self.card)


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # The last line can be cut short if recording was killed.
                continue
    return rows


def load_recording(path: str) -> Recording:
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as file:
        meta = json.load(file)
    deck = [card_by_name(name) for name in meta["deck"]]

    times, states = [], []
    for row in _read_jsonl(os.path.join(path, "states.jsonl")):
        times.append(float(row["t"]))
        states.append(state_from_dict(row["state"]))

    events = []
    for row in _read_jsonl(os.path.join(path, "touch.jsonl")):
        event = parse_event_line(row["line"])
        if event is not None:
            events.append((float(row["t"]), event))

    plays = []
    if events:
        # getevent stamps events with the device's clock. The smallest gap
        # between when a line arrived and its stamp is the best estimate
        # of the offset between the two clocks.
        offset = min(t - e.device_time for t, e in events)
        builder = GestureBuilder(
            meta.get("touch_ranges", {}),
            Calibration.from_config(meta.get("calibration")),
        )
        for _, event in events:
            builder.feed(event)
        plays = [
            Play(
                p.t_select + offset,
                p.t_place + offset,
                p.slot,
                p.tile_x,
                p.tile_y,
            )
            for p in gestures_to_plays(builder.gestures)
        ]

    order = np.argsort(times, kind="stable")
    times = [times[i] for i in order]
    states = [states[i] for i in order]
    return Recording(path, deck, times, states, plays)


def find_recordings(root: str) -> list[str]:
    found = []
    for dirpath, _, filenames in os.walk(root):
        if "meta.json" in filenames:
            found.append(dirpath)
    return sorted(found)


def _confirmed(rec: Recording, play: Play, card) -> bool:
    """Did the card leave that hand position soon after it was placed?"""
    start = bisect.bisect_right(rec.times, play.t_place)
    end = bisect.bisect_right(rec.times, play.t_place + CONFIRM_TIME)
    for i in range(start, end):
        cards = rec.states[i].cards
        if (
            rec.states[i].screen == Screens.IN_GAME
            and len(cards) > play.slot + 1
        ):
            if cards[play.slot + 1].name != card.name:
                return True
    return False


def label_recording(rec: Recording, deck, stats: Counter):
    """Yield (state_index, deck_index or WAIT, tile_x, tile_y)."""
    encoder = FeatureEncoder(deck)
    labels: dict[int, tuple[int, int, int]] = {}
    blocked: set[int] = set()

    for play in rec.plays:
        stats["taps_detected"] += 1
        first = bisect.bisect_left(rec.times, play.t_select - LEAD_TIME)
        decision = bisect.bisect_right(rec.times, play.t_select) - 1
        last_blocked = bisect.bisect_right(rec.times, play.t_place + 0.3)
        # Never label the moments around a play as "wait", even if the
        # play itself gets skipped below.
        blocked.update(range(max(first, 0), last_blocked))

        if decision < 0:
            stats["skipped_before_first_frame"] += 1
            continue
        state = rec.states[decision]
        if (
            state.screen != Screens.IN_GAME
            or len(state.cards) <= play.slot + 1
        ):
            stats["skipped_not_in_battle"] += 1
            continue
        card = state.cards[play.slot + 1]
        index = encoder.deck_index(card)
        if index is None:
            stats["skipped_card_not_recognised"] += 1
            continue
        if not _confirmed(rec, play, card):
            stats["skipped_not_confirmed"] += 1
            continue

        stats["plays"] += 1
        for i in range(max(first, 0), decision + 1):
            s = rec.states[i]
            if (
                s.screen == Screens.IN_GAME
                and len(s.cards) > play.slot + 1
                and s.cards[play.slot + 1].name == card.name
            ):
                labels[i] = (index, play.tile_x, play.tile_y)

    for i, state in enumerate(rec.states):
        if state.screen != Screens.IN_GAME:
            continue
        if i in labels:
            yield i, *labels[i]
        elif i not in blocked and encoder.playable(state):
            # States with nothing playable teach the model nothing.
            yield i, WAIT, -1, -1


def build_examples(recording_dirs: list[str], deck=None) -> Examples:
    stats: Counter = Counter()
    rows = []
    for rec_i, path in enumerate(recording_dirs):
        rec = load_recording(path)
        if deck is None:
            deck = rec.deck
        if [c.name for c in rec.deck] != [c.name for c in deck]:
            stats["recordings_skipped_other_deck"] += 1
            continue
        stats["recordings"] += 1
        encoder = FeatureEncoder(deck)
        for i, card, tile_x, tile_y in label_recording(rec, deck, stats):
            state = rec.states[i]
            grid, vec = encoder.encode(state)
            card_mask = encoder.card_mask(state)
            tile_masks = encoder.tile_masks(state)
            if card != WAIT:
                # Trust the player over a slow elixir/ready reading.
                card_mask[card] = True
                snapped = nearest_allowed_tile(
                    tile_masks[card], tile_x, tile_y
                )
                if snapped is None:
                    continue
                tile_x, tile_y = snapped
                stats["examples_play"] += 1
            else:
                stats["examples_wait"] += 1
            group = rec_i * 100_000 + int(rec.times[i] // CHUNK_SECONDS)
            rows.append(
                (grid, vec, card_mask, tile_masks, card, tile_x, tile_y, group)
            )

    if deck is None:
        raise ValueError("No recordings found.")
    if not rows:
        raise ValueError(
            "The recordings have no usable battle moments. "
            f"Details: {dict(stats)}"
        )
    grid, vec, card_mask, tile_masks, card, tx, ty, group = zip(
        *rows, strict=True
    )
    examples = Examples(
        grid=np.clip(np.stack(grid), 0, 255).astype(np.uint8),
        vec=np.stack(vec).astype(np.float32),
        card_mask=np.stack(card_mask),
        tile_masks=np.stack(tile_masks),
        card=np.array(card, dtype=np.int64),
        tile_x=np.array(tx, dtype=np.int64),
        tile_y=np.array(ty, dtype=np.int64),
        group=np.array(group, dtype=np.int64),
        deck=deck,
        stats=stats,
    )
    return examples
