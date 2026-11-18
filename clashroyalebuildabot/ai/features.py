"""Turn a State into the arrays the policy network reads.

The training script and the bot both use this, so a model always sees
exactly the same kind of input it was trained on.

grid: (GRID_CHANNELS, 32, 18), indexed [channel, tile_y, tile_x], where
      tile_y 0 is the row nearest your king tower.
vec:  everything that isn't a position: elixir, tower HP, the hand, the
      next card and how many of each unit are on the field.
"""

import numpy as np

from clashroyalebuildabot.ai.touch import N_TALL_TILES
from clashroyalebuildabot.constants import (
    ALL_TILES,
    ALLY_TILES,
    DECK_SIZE,
    DETECTOR_UNITS,
    LEFT_PRINCESS_TILES,
    N_WIDE_TILES,
    RIGHT_PRINCESS_TILES,
)
from clashroyalebuildabot.namespaces import State
from clashroyalebuildabot.namespaces.cards import Card
from clashroyalebuildabot.namespaces.units import (
    Target,
    Transport,
    UnitCategory,
)

FEATURE_VERSION = 1
GRID_SHAPE = (N_TALL_TILES, N_WIDE_TILES)

# Per side (ally, enemy): every unit, ground troops, air troops,
# buildings, troops that only hit buildings. Then the playable tiles.
_SIDE_CHANNELS = 5
VALID_CHANNEL = 2 * _SIDE_CHANNELS
GRID_CHANNELS = VALID_CHANNEL + 1

# vec layout
VEC_ELIXIR = 0
# left enemy, right enemy, left ally, right ally princess HP (0-1)
VEC_HP = slice(1, 5)
_HAND_FEATURES = DECK_SIZE + 2  # which card, ready, affordable
VEC_HAND = slice(5, 5 + 4 * _HAND_FEATURES)
VEC_NEXT = slice(VEC_HAND.stop, VEC_HAND.stop + DECK_SIZE)
_UNIT_INDEX = {unit.name: i for i, unit in enumerate(DETECTOR_UNITS)}
VEC_UNITS = slice(VEC_NEXT.stop, VEC_NEXT.stop + 2 * len(DETECTOR_UNITS))
VEC_SIZE = VEC_UNITS.stop

WAIT = DECK_SIZE  # index of "don't play" in the card choice
N_CHOICES = DECK_SIZE + 1


def _tile_mask(tiles) -> np.ndarray:
    mask = np.zeros(GRID_SHAPE, dtype=bool)
    for x, y in tiles:
        mask[y, x] = True
    return mask


ALL_TILES_MASK = _tile_mask(ALL_TILES)
_ALLY_MASK = _tile_mask(ALLY_TILES)
_LEFT_MASK = _tile_mask(LEFT_PRINCESS_TILES)
_RIGHT_MASK = _tile_mask(RIGHT_PRINCESS_TILES)


def valid_tiles_mask(state: State) -> np.ndarray:
    """Where a troop or building can go (matches Bot._get_valid_tiles)."""
    mask = _ALLY_MASK.copy()
    if state.numbers.left_enemy_princess_hp.number == 0:
        mask |= _LEFT_MASK
    if state.numbers.right_enemy_princess_hp.number == 0:
        mask |= _RIGHT_MASK
    return mask


class FeatureEncoder:
    def __init__(self, deck: list[Card]):
        if len(deck) != DECK_SIZE:
            raise ValueError(
                f"A deck needs {DECK_SIZE} cards, got {len(deck)}"
            )
        self.deck = list(deck)
        self._deck_index = {card.name: i for i, card in enumerate(deck)}

    def deck_index(self, card: Card) -> int | None:
        return self._deck_index.get(card.name)

    def encode(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        grid = np.zeros((GRID_CHANNELS, *GRID_SHAPE), dtype=np.float32)
        vec = np.zeros(VEC_SIZE, dtype=np.float32)

        for side, detections in enumerate((state.allies, state.enemies)):
            base = side * _SIDE_CHANNELS
            for det in detections:
                x = min(max(det.position.tile_x, 0), N_WIDE_TILES - 1)
                y = min(max(det.position.tile_y, 0), N_TALL_TILES - 1)
                unit = det.unit
                grid[base, y, x] += 1
                if unit.category == UnitCategory.BUILDING:
                    grid[base + 3, y, x] += 1
                elif unit.transport == Transport.AIR:
                    grid[base + 2, y, x] += 1
                else:
                    grid[base + 1, y, x] += 1
                if unit.target == Target.BUILDINGS:
                    grid[base + 4, y, x] += 1

                index = _UNIT_INDEX.get(unit.name)
                if index is not None:
                    offset = VEC_UNITS.start + side * len(DETECTOR_UNITS)
                    vec[offset + index] += 1
        grid[VALID_CHANNEL] = valid_tiles_mask(state)

        units = vec[VEC_UNITS]
        vec[VEC_UNITS] = np.minimum(units, 5) / 5

        elixir = float(state.numbers.elixir.number)
        vec[VEC_ELIXIR] = elixir / 10
        numbers = state.numbers
        vec[VEC_HP] = [
            numbers.left_enemy_princess_hp.number,
            numbers.right_enemy_princess_hp.number,
            numbers.left_ally_princess_hp.number,
            numbers.right_ally_princess_hp.number,
        ]

        for slot in range(4):
            if slot + 1 >= len(state.cards):
                break
            card = state.cards[slot + 1]
            offset = VEC_HAND.start + slot * _HAND_FEATURES
            index = self.deck_index(card)
            if index is not None:
                vec[offset + index] = 1
            vec[offset + DECK_SIZE] = slot in state.ready
            vec[offset + DECK_SIZE + 1] = elixir >= card.cost
        if state.cards:
            index = self.deck_index(state.cards[0])
            if index is not None:
                vec[VEC_NEXT.start + index] = 1

        return grid, vec

    def playable(self, state: State) -> dict[int, int]:
        """Deck index -> hand slot, for cards that can be played now."""
        elixir = state.numbers.elixir.number
        result = {}
        for slot in state.ready:
            if slot + 1 >= len(state.cards):
                continue
            card = state.cards[slot + 1]
            index = self.deck_index(card)
            if index is not None and elixir >= card.cost:
                result[index] = slot
        return result

    def card_mask(self, state: State) -> np.ndarray:
        mask = np.zeros(N_CHOICES, dtype=bool)
        mask[list(self.playable(state))] = True
        mask[WAIT] = True
        return mask

    def tile_masks(self, state: State) -> np.ndarray:
        """(DECK_SIZE, 32, 18): where each deck card may be placed."""
        valid = valid_tiles_mask(state)
        return np.stack(
            [
                ALL_TILES_MASK if card.target_anywhere else valid
                for card in self.deck
            ]
        )


def flip_horizontal(grid, vec, tile_masks, tile_x):
    """Mirror the arena left-to-right, which is an equally good game."""
    grid = grid[..., ::-1].copy()
    tile_masks = tile_masks[..., ::-1].copy()
    vec = vec.copy()
    hp = vec[..., VEC_HP].copy()
    vec[..., VEC_HP.start] = hp[..., 1]
    vec[..., VEC_HP.start + 1] = hp[..., 0]
    vec[..., VEC_HP.start + 2] = hp[..., 3]
    vec[..., VEC_HP.start + 3] = hp[..., 2]
    return grid, vec, tile_masks, N_WIDE_TILES - 1 - tile_x


def nearest_allowed_tile(mask: np.ndarray, tile_x: int, tile_y: int):
    """The allowed tile closest to (tile_x, tile_y), or None if none are."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    best = np.argmin((xs - tile_x) ** 2 + (ys - tile_y) ** 2)
    return int(xs[best]), int(ys[best])
