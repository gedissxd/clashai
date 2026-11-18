"""Read the player's taps from ``adb shell getevent`` and map them to plays.

A card can be played two ways: tap the card then tap the arena, or drag
the card onto the arena. Both are turned into a ``Play``.
"""

import math
import re
from dataclasses import dataclass

from clashroyalebuildabot.constants import (
    DISPLAY_CARD_DELTA_X,
    DISPLAY_CARD_HEIGHT,
    DISPLAY_CARD_INIT_X,
    DISPLAY_CARD_WIDTH,
    DISPLAY_CARD_Y,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    N_WIDE_TILES,
    TILE_HEIGHT,
    TILE_INIT_X,
    TILE_INIT_Y,
    TILE_WIDTH,
)

N_TALL_TILES = 32
# A selected card stays selected for this long waiting for an arena tap.
SELECT_TIMEOUT = 5.0
# A touch that moves less than this (in 720x1280 pixels) is a tap.
TAP_MAX_MOVE = 25

_RANGE_RE = re.compile(
    r"(ABS_(?:MT_POSITION_X|MT_POSITION_Y|X|Y))\s*:\s*value\s+-?\d+,"
    r"\s*min\s+(-?\d+),\s*max\s+(-?\d+)"
)
_DEVICE_RE = re.compile(r"add device \d+:\s*(\S+)")
_EVENT_RE = re.compile(
    r"^\[\s*([\d.]+)\]\s+(\S+):\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)


def parse_ranges(getevent_p_output: str) -> dict[str, dict[str, list[int]]]:
    """Axis ranges per input device, from ``getevent -lp``."""
    ranges: dict[str, dict[str, list[int]]] = {}
    device = None
    for line in getevent_p_output.splitlines():
        match = _DEVICE_RE.search(line)
        if match:
            device = match.group(1)
            continue
        match = _RANGE_RE.search(line)
        if match and device is not None:
            axis, lo, hi = (
                match.group(1),
                int(match.group(2)),
                int(match.group(3)),
            )
            ranges.setdefault(device, {})[axis] = [lo, hi]
    return ranges


@dataclass(frozen=True)
class RawEvent:
    device_time: float
    device: str
    type: str
    code: str
    value: str


def parse_event_line(line: str) -> RawEvent | None:
    """Parse one line of ``getevent -lt``; None for anything else."""
    match = _EVENT_RE.match(line.strip())
    if not match:
        return None
    t, device, type_, code, value = match.groups()
    return RawEvent(float(t), device.rstrip(":"), type_, code, value)


@dataclass(frozen=True)
class Gesture:
    """One finger going down and coming up, in 720x1280 coordinates."""

    t_down: float
    t_up: float
    x_down: float
    y_down: float
    x_up: float
    y_up: float

    @property
    def is_tap(self) -> bool:
        moved = math.hypot(self.x_up - self.x_down, self.y_up - self.y_down)
        return moved <= TAP_MAX_MOVE


@dataclass(frozen=True)
class Calibration:
    """Fixes for emulators whose touch axes don't match the screen."""

    swap_xy: bool = False
    flip_x: bool = False
    flip_y: bool = False

    @classmethod
    def from_config(cls, config: dict | None) -> "Calibration":
        config = config or {}
        return cls(
            swap_xy=bool(config.get("swap_xy", False)),
            flip_x=bool(config.get("flip_x", False)),
            flip_y=bool(config.get("flip_y", False)),
        )

    def apply(self, nx: float, ny: float) -> tuple[float, float]:
        """Normalised (0-1) touch position -> 720x1280 screen position."""
        if self.swap_xy:
            nx, ny = ny, nx
        if self.flip_x:
            nx = 1 - nx
        if self.flip_y:
            ny = 1 - ny
        return nx * DISPLAY_WIDTH, ny * DISPLAY_HEIGHT


def _normalise(raw: int, axis_range: list[int] | None) -> float:
    if not axis_range or axis_range[1] <= axis_range[0]:
        return 0.0
    lo, hi = axis_range
    return min(max((raw - lo) / (hi - lo), 0.0), 1.0)


class GestureBuilder:
    """Feed events in order; collects finished single-finger gestures."""

    def __init__(self, ranges, calibration: Calibration):
        self.ranges = ranges
        self.calibration = calibration
        self.gestures: list[Gesture] = []
        self._dev: dict[str, dict] = {}

    def _device_state(self, device):
        if device not in self._dev:
            self._dev[device] = {
                "slot": 0,
                "raw_x": None,
                "raw_y": None,
                "pending_down": False,
                "pending_up": False,
                "down": None,  # (t, x, y) while touching
                "last_xy": None,
            }
        return self._dev[device]

    def _xy(self, device, st):
        axes = self.ranges.get(device, {})
        x_range = axes.get("ABS_MT_POSITION_X") or axes.get("ABS_X")
        y_range = axes.get("ABS_MT_POSITION_Y") or axes.get("ABS_Y")
        nx = _normalise(st["raw_x"], x_range)
        ny = _normalise(st["raw_y"], y_range)
        return self.calibration.apply(nx, ny)

    def feed(self, event: RawEvent):
        st = self._device_state(event.device)
        code, value = event.code, event.value

        if code == "ABS_MT_SLOT":
            st["slot"] = int(value, 16)
            return
        # Only follow the first finger.
        if st["slot"] != 0 and code.startswith("ABS_MT"):
            return

        if code in ("ABS_MT_POSITION_X", "ABS_X"):
            st["raw_x"] = int(value, 16)
        elif code in ("ABS_MT_POSITION_Y", "ABS_Y"):
            st["raw_y"] = int(value, 16)
        elif code == "ABS_MT_TRACKING_ID":
            if value.lower() == "ffffffff":
                st["pending_up"] = True
            else:
                st["pending_down"] = True
        elif code == "BTN_TOUCH":
            if value == "DOWN":
                st["pending_down"] = True
            elif value == "UP":
                st["pending_up"] = True
        elif code == "SYN_REPORT":
            self._commit(event, st)

    def _commit(self, event, st):
        if st["raw_x"] is not None and st["raw_y"] is not None:
            xy = self._xy(event.device, st)
        else:
            xy = None

        if st["pending_down"] and st["down"] is None and xy is not None:
            st["down"] = (event.device_time, *xy)
        if xy is not None:
            st["last_xy"] = xy
        if st["pending_up"] and st["down"] is not None:
            t0, x0, y0 = st["down"]
            x1, y1 = st["last_xy"] or (x0, y0)
            self.gestures.append(
                Gesture(t0, event.device_time, x0, y0, x1, y1)
            )
            st["down"] = None
        st["pending_down"] = False
        st["pending_up"] = False


def card_slot_at(x: float, y: float) -> int | None:
    """Hand position (0-3) under a screen point, if any."""
    if not DISPLAY_CARD_Y <= y <= DISPLAY_CARD_Y + DISPLAY_CARD_HEIGHT:
        return None
    for i in range(4):
        left = DISPLAY_CARD_INIT_X + i * DISPLAY_CARD_DELTA_X
        if left <= x <= left + DISPLAY_CARD_WIDTH:
            return i
    return None


def tile_at(x: float, y: float) -> tuple[int, int] | None:
    """Arena tile under a screen point, if any (inverse of the bot's taps)."""
    tile_x = math.floor((x - TILE_INIT_X) / TILE_WIDTH)
    tile_y = math.floor((DISPLAY_HEIGHT - TILE_INIT_Y - y) / TILE_HEIGHT)
    if 0 <= tile_x < N_WIDE_TILES and 0 <= tile_y < N_TALL_TILES:
        return tile_x, tile_y
    return None


@dataclass(frozen=True)
class Play:
    """The player picked hand position ``slot`` and dropped it on a tile."""

    t_select: float
    t_place: float
    slot: int
    tile_x: int
    tile_y: int


def gestures_to_plays(gestures: list[Gesture]) -> list[Play]:
    plays = []
    selected: tuple[float, int] | None = None
    for g in gestures:
        slot = card_slot_at(g.x_down, g.y_down)
        if slot is not None:
            drop = tile_at(g.x_up, g.y_up)
            if not g.is_tap and drop is not None:
                # Dragged a card onto the arena.
                plays.append(Play(g.t_down, g.t_up, slot, *drop))
                selected = None
            else:
                selected = (g.t_down, slot)
            continue

        tile = tile_at(g.x_down, g.y_down)
        if (
            g.is_tap
            and tile is not None
            and selected is not None
            and g.t_down - selected[0] <= SELECT_TIMEOUT
        ):
            plays.append(Play(selected[0], g.t_down, selected[1], *tile))
        selected = None
    return plays
