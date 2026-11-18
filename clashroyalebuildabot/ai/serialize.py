"""Turn a State into plain JSON and back, so recordings can be replayed."""

from clashroyalebuildabot.namespaces import Cards, Screens, State, Units
from clashroyalebuildabot.namespaces.cards import Card
from clashroyalebuildabot.namespaces.numbers import NumberDetection, Numbers
from clashroyalebuildabot.namespaces.screens import Screen
from clashroyalebuildabot.namespaces.units import (
    Position,
    Unit,
    UnitDetection,
)

CARDS_BY_NAME: dict[str, Card] = {
    card.name: card for card in vars(Cards).values() if isinstance(card, Card)
}
UNITS_BY_NAME: dict[str, Unit] = {
    unit.name: unit for unit in vars(Units).values() if isinstance(unit, Unit)
}
SCREENS_BY_NAME: dict[str, Screen] = {
    screen.name: screen
    for screen in vars(Screens).values()
    if isinstance(screen, Screen)
}
NUMBER_FIELDS = (
    "left_enemy_princess_hp",
    "right_enemy_princess_hp",
    "left_ally_princess_hp",
    "right_ally_princess_hp",
    "elixir",
)


def card_by_name(name: str) -> Card:
    try:
        return CARDS_BY_NAME[name]
    except KeyError:
        raise ValueError(f"Unknown card {name!r}") from None


def _unit_to_list(detection: UnitDetection) -> list:
    pos = detection.position
    return [
        detection.unit.name,
        pos.tile_x,
        pos.tile_y,
        round(float(pos.conf), 3),
        [int(v) for v in pos.bbox],
    ]


def _unit_from_list(item: list) -> UnitDetection | None:
    name, tile_x, tile_y, conf, bbox = item
    unit = UNITS_BY_NAME.get(name)
    if unit is None:
        return None
    return UnitDetection(
        unit, Position(tuple(bbox), conf, int(tile_x), int(tile_y))
    )


def state_to_dict(state: State) -> dict:
    return {
        "allies": [_unit_to_list(u) for u in state.allies],
        "enemies": [_unit_to_list(u) for u in state.enemies],
        "numbers": {
            field: float(getattr(state.numbers, field).number)
            for field in NUMBER_FIELDS
        },
        "cards": [card.name for card in state.cards],
        "ready": [int(i) for i in state.ready],
        "screen": state.screen.name,
    }


def state_from_dict(data: dict) -> State:
    bbox = (0, 0, 0, 0)
    numbers = Numbers(
        **{
            field: NumberDetection(bbox, data["numbers"][field])
            for field in NUMBER_FIELDS
        }
    )
    allies = [_unit_from_list(u) for u in data["allies"]]
    enemies = [_unit_from_list(u) for u in data["enemies"]]
    return State(
        allies=[u for u in allies if u is not None],
        enemies=[u for u in enemies if u is not None],
        numbers=numbers,
        cards=[CARDS_BY_NAME.get(name, Cards.BLANK) for name in data["cards"]],
        ready=list(data["ready"]),
        screen=SCREENS_BY_NAME.get(data["screen"], Screens.UNKNOWN),
    )
