import json

import numpy as np
import pytest
from test_ai_features import DECK
from test_ai_touch import GETEVENT_P, touch_lines

from clashroyalebuildabot.ai.dataset import build_examples, find_recordings
from clashroyalebuildabot.ai.features import WAIT
from clashroyalebuildabot.ai.serialize import state_to_dict
from clashroyalebuildabot.ai.touch import parse_ranges
from clashroyalebuildabot.bot.bot import Bot
from clashroyalebuildabot.namespaces import Cards, Screens

# getevent's clock runs this far behind the computer's in the fake recording
CLOCK_OFFSET = 50.0

HAND = [
    Cards.WITCH,
    Cards.ARCHERS,
    Cards.GOBLIN_BARREL,
    Cards.BABY_DRAGON,
    Cards.CANNON,
]
AFTER_ARCHERS = [
    Cards.KNIGHT,
    Cards.WITCH,
    Cards.GOBLIN_BARREL,
    Cards.BABY_DRAGON,
    Cards.CANNON,
]


def write_recording(path, state_factory):
    path.mkdir()
    meta = {
        "version": 1,
        "deck": [card.name for card in DECK],
        "touch_ranges": parse_ranges(GETEVENT_P),
        "calibration": {},
    }
    (path / "meta.json").write_text(json.dumps(meta))

    with open(path / "states.jsonl", "w") as f:
        for i in range(60):
            t = 100.0 + i * 0.2
            # Archers leave hand position 0 just after being placed at 104.5
            cards = HAND if t < 104.7 else AFTER_ARCHERS
            screen = Screens.IN_GAME if t >= 100.4 else Screens.LOBBY
            state = state_factory(
                elixir=6, cards=cards, ready=[0, 1, 2, 3], screen=screen
            )
            f.write(json.dumps({"t": t, "state": state_to_dict(state)}) + "\n")
        f.write('{"t": 200, "sta')  # cut short when recording stopped

    lines = []
    # Archers (hand position 0) tapped, then tile (5, 7)
    lines += touch_lines(104.0 - CLOCK_OFFSET, *Bot._get_card_centre(0))
    lines += touch_lines(104.5 - CLOCK_OFFSET, *Bot._get_tile_centre(5, 7))
    # Goblin barrel tapped and placed, but it never leaves the hand
    lines += touch_lines(108.0 - CLOCK_OFFSET, *Bot._get_card_centre(1))
    lines += touch_lines(108.3 - CLOCK_OFFSET, *Bot._get_tile_centre(9, 25))
    with open(path / "touch.jsonl", "w") as f:
        for line in lines:
            device_time = float(line[1:16])
            # lines arrive a little after they happen
            row = {"t": device_time + CLOCK_OFFSET + 0.02, "line": line}
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def examples(tmp_path, state_factory):
    write_recording(tmp_path / "rec1", state_factory)
    dirs = find_recordings(str(tmp_path))
    assert len(dirs) == 1
    return build_examples(dirs)


def test_confirmed_play_is_labelled(examples):
    assert examples.stats["taps_detected"] == 2
    assert examples.stats["plays"] == 1
    assert examples.stats["skipped_not_confirmed"] == 1

    plays = examples.card != WAIT
    assert 3 <= plays.sum() <= 4
    assert set(examples.card[plays]) == {0}  # archers
    assert set(
        zip(examples.tile_x[plays], examples.tile_y[plays], strict=True)
    ) == {(5, 7)}


def test_wait_examples_skip_lobby_and_moments_around_plays(examples):
    waits = examples.card == WAIT
    assert waits.sum() > 20
    assert (examples.tile_x[waits] == -1).all()
    n_states_in_battle = 58
    assert len(examples) < n_states_in_battle


def test_arrays_line_up(examples):
    n = len(examples)
    assert examples.grid.dtype == np.uint8
    for array in (
        examples.vec,
        examples.card_mask,
        examples.tile_masks,
        examples.tile_x,
        examples.group,
    ):
        assert len(array) == n
    assert [c.name for c in examples.deck] == [c.name for c in DECK]
