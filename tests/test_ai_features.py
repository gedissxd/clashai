import numpy as np
import pytest

from clashroyalebuildabot.ai.features import (
    GRID_CHANNELS,
    GRID_SHAPE,
    VEC_ELIXIR,
    VEC_SIZE,
    WAIT,
    FeatureEncoder,
    flip_horizontal,
    nearest_allowed_tile,
)
from clashroyalebuildabot.ai.policy import choose
from clashroyalebuildabot.ai.serialize import state_from_dict, state_to_dict
from clashroyalebuildabot.namespaces import Cards, Units

DECK = [
    Cards.ARCHERS,
    Cards.GOBLIN_BARREL,
    Cards.BABY_DRAGON,
    Cards.CANNON,
    Cards.KNIGHT,
    Cards.MINIPEKKA,
    Cards.MUSKETEER,
    Cards.WITCH,
]


@pytest.fixture
def battle_state(state_factory, unit_factory):
    return state_factory(
        allies=[unit_factory(3, 4, Units.KNIGHT)],
        enemies=[
            unit_factory(14, 20, Units.HOG_RIDER),
            unit_factory(14, 20, Units.MINION),
        ],
        elixir=4,
        left_enemy_hp=0.0,
        right_enemy_hp=0.7,
        left_ally_hp=0.9,
        right_ally_hp=0.4,
        cards=[
            Cards.WITCH,
            Cards.ARCHERS,
            Cards.GOBLIN_BARREL,
            Cards.BABY_DRAGON,
            Cards.CANNON,
        ],
        ready=[0, 1, 2, 3],
    )


def test_encode_shapes_and_values(battle_state):
    encoder = FeatureEncoder(DECK)
    grid, vec = encoder.encode(battle_state)
    assert grid.shape == (GRID_CHANNELS, *GRID_SHAPE)
    assert vec.shape == (VEC_SIZE,)
    assert vec[VEC_ELIXIR] == pytest.approx(0.4)
    # ally knight: every-unit and ground channels
    assert grid[0, 4, 3] == 1 and grid[1, 4, 3] == 1
    # enemy hog + minion on the same tile
    assert grid[5, 20, 14] == 2
    # left enemy tower is down, so its side is playable
    assert grid[-1, 18, 2] == 1 and grid[-1, 18, 15] == 0


def test_playable_respects_elixir(battle_state):
    encoder = FeatureEncoder(DECK)
    # 4 elixir: archers (3), goblin barrel (3), baby dragon (4), cannon (3)
    assert encoder.playable(battle_state) == {0: 0, 1: 1, 2: 2, 3: 3}
    mask = encoder.card_mask(battle_state)
    assert mask[WAIT] and mask[:4].all() and not mask[4:WAIT].any()


def test_spells_can_target_anywhere(battle_state):
    masks = FeatureEncoder(DECK).tile_masks(battle_state)
    goblin_barrel, knight = masks[1], masks[4]
    assert goblin_barrel[25, 9]
    assert not knight[25, 9]


def test_flip_matches_encoding_the_mirrored_state(
    battle_state, state_factory, unit_factory
):
    mirrored = state_factory(
        allies=[unit_factory(14, 4, Units.KNIGHT)],
        enemies=[
            unit_factory(3, 20, Units.HOG_RIDER),
            unit_factory(3, 20, Units.MINION),
        ],
        elixir=4,
        left_enemy_hp=0.7,
        right_enemy_hp=0.0,
        left_ally_hp=0.4,
        right_ally_hp=0.9,
        cards=battle_state.cards,
        ready=battle_state.ready,
    )
    encoder = FeatureEncoder(DECK)
    grid, vec = encoder.encode(battle_state)
    masks = encoder.tile_masks(battle_state)
    f_grid, f_vec, f_masks, f_x = flip_horizontal(grid, vec, masks, 2)

    m_grid, m_vec = encoder.encode(mirrored)
    np.testing.assert_array_equal(f_grid, m_grid)
    np.testing.assert_array_equal(f_vec, m_vec)
    np.testing.assert_array_equal(f_masks, encoder.tile_masks(mirrored))
    assert f_x == 15


def test_state_round_trips_through_json(battle_state):
    again = state_from_dict(state_to_dict(battle_state))
    assert state_to_dict(again) == state_to_dict(battle_state)
    assert again.enemies[0].unit == Units.HOG_RIDER
    assert again.cards[0] == Cards.WITCH


def test_nearest_allowed_tile_snaps():
    mask = np.zeros(GRID_SHAPE, dtype=bool)
    mask[3, 4] = True
    assert nearest_allowed_tile(mask, 5, 5) == (4, 3)
    assert nearest_allowed_tile(np.zeros(GRID_SHAPE, dtype=bool), 0, 0) is None


def test_choose_plays_only_when_confident(battle_state):
    encoder = FeatureEncoder(DECK)
    card_mask = encoder.card_mask(battle_state)
    tile_masks = encoder.tile_masks(battle_state)
    playable = encoder.playable(battle_state)
    tile_logits = np.zeros((8, *GRID_SHAPE), dtype=np.float32)
    tile_logits[2, 6, 9] = 5  # baby dragon goes on (9, 6)
    tile_logits[2, 25, 9] = 9  # ...not in the enemy half, which isn't allowed

    card_logits = np.zeros(9, dtype=np.float32)
    card_logits[2] = 3  # baby dragon
    card_logits[5] = 10  # mini pekka: not affordable, must be ignored
    choice = choose(
        card_logits, tile_logits, card_mask, tile_masks, playable, 0.5
    )
    assert choice[:3] == (2, 9, 6)

    card_logits[WAIT] = 5
    assert (
        choose(card_logits, tile_logits, card_mask, tile_masks, playable, 0.5)
        is None
    )
