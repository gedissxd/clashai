import pytest

from clashroyalebuildabot.ai.touch import (
    Calibration,
    GestureBuilder,
    Play,
    card_slot_at,
    gestures_to_plays,
    parse_event_line,
    parse_ranges,
    tile_at,
)
from clashroyalebuildabot.bot.bot import Bot

GETEVENT_P = """\
add device 1: /dev/input/event3
  name:     "Power Button"
add device 2: /dev/input/event4
  name:     "Virtual Touch"
  events:
    ABS (0003): ABS_MT_SLOT           : value 0, min 0, max 9, fuzz 0, flat 0, resolution 0
                ABS_MT_POSITION_X     : value 0, min 0, max 32767, fuzz 0, flat 0, resolution 0
                ABS_MT_POSITION_Y     : value 0, min 0, max 32767, fuzz 0, flat 0, resolution 0
                ABS_MT_TRACKING_ID    : value 0, min 0, max 65535, fuzz 0, flat 0, resolution 0
"""
DEVICE = "/dev/input/event4"
RANGES = parse_ranges(GETEVENT_P)


def raw(x, y):
    """720x1280 screen position -> raw 0-32767 touch values."""
    return round(x / 720 * 32767), round(y / 1280 * 32767)


def touch_lines(t, x, y, t_up=None, x_up=None, y_up=None):
    """getevent -lt lines for one finger going down and up."""
    rx, ry = raw(x, y)
    t_up = t + 0.05 if t_up is None else t_up
    ux, uy = raw(x if x_up is None else x_up, y if y_up is None else y_up)
    dev = DEVICE
    return [
        f"[{t:15.6f}] {dev}: EV_ABS       ABS_MT_TRACKING_ID   00000001",
        f"[{t:15.6f}] {dev}: EV_ABS       ABS_MT_POSITION_X    {rx:08x}",
        f"[{t:15.6f}] {dev}: EV_ABS       ABS_MT_POSITION_Y    {ry:08x}",
        f"[{t:15.6f}] {dev}: EV_KEY       BTN_TOUCH            DOWN",
        f"[{t:15.6f}] {dev}: EV_SYN       SYN_REPORT           00000000",
        f"[{t_up:15.6f}] {dev}: EV_ABS       ABS_MT_POSITION_X    {ux:08x}",
        f"[{t_up:15.6f}] {dev}: EV_ABS       ABS_MT_POSITION_Y    {uy:08x}",
        f"[{t_up:15.6f}] {dev}: EV_SYN       SYN_REPORT           00000000",
        f"[{t_up:15.6f}] {dev}: EV_ABS       ABS_MT_TRACKING_ID   ffffffff",
        f"[{t_up:15.6f}] {dev}: EV_KEY       BTN_TOUCH            UP",
        f"[{t_up:15.6f}] {dev}: EV_SYN       SYN_REPORT           00000000",
    ]


def gestures_from(lines, calibration=None):
    builder = GestureBuilder(RANGES, calibration or Calibration())
    for line in lines:
        event = parse_event_line(line)
        assert event is not None, line
        builder.feed(event)
    return builder.gestures


def test_parse_ranges_finds_touch_axes():
    assert RANGES == {
        DEVICE: {
            "ABS_MT_POSITION_X": [0, 32767],
            "ABS_MT_POSITION_Y": [0, 32767],
        }
    }


def test_parse_event_line_ignores_other_output():
    assert parse_event_line("could not open /dev/input/event9") is None
    event = parse_event_line(
        "[   1234.500000] /dev/input/event4: EV_ABS ABS_MT_POSITION_X 0000000a"
    )
    assert event.device_time == 1234.5
    assert event.code == "ABS_MT_POSITION_X"


def test_tap_position_is_scaled_to_the_screen():
    (gesture,) = gestures_from(touch_lines(10.0, 360, 640))
    assert gesture.is_tap
    assert gesture.x_down == pytest.approx(360, abs=1)
    assert gesture.y_down == pytest.approx(640, abs=1)


def test_calibration_swaps_and_flips():
    cal = Calibration(swap_xy=True, flip_x=True)
    assert cal.apply(0.25, 0.75) == pytest.approx((0.25 * 720, 0.25 * 1280))


@pytest.mark.parametrize("slot", range(4))
def test_card_slot_matches_where_the_bot_taps(slot):
    assert card_slot_at(*Bot._get_card_centre(slot)) == slot


@pytest.mark.parametrize("tile", [(0, 0), (8, 5), (17, 14), (3, 20)])
def test_tile_matches_where_the_bot_taps(tile):
    assert tile_at(*Bot._get_tile_centre(*tile)) == tile


def test_tap_card_then_tap_arena_is_a_play():
    card_xy = Bot._get_card_centre(2)
    tile_xy = Bot._get_tile_centre(5, 7)
    lines = touch_lines(10.0, *card_xy) + touch_lines(11.0, *tile_xy)
    (play,) = gestures_to_plays(gestures_from(lines))
    assert play == Play(10.0, 11.0, 2, 5, 7)


def test_drag_card_onto_arena_is_a_play():
    card_xy = Bot._get_card_centre(0)
    tile_xy = Bot._get_tile_centre(12, 3)
    lines = touch_lines(
        10.0, *card_xy, t_up=10.4, x_up=tile_xy[0], y_up=tile_xy[1]
    )
    (play,) = gestures_to_plays(gestures_from(lines))
    assert (play.slot, play.tile_x, play.tile_y) == (0, 12, 3)
    assert play.t_place == pytest.approx(10.4)


def test_arena_tap_without_a_card_is_not_a_play():
    lines = touch_lines(10.0, *Bot._get_tile_centre(5, 7))
    assert gestures_to_plays(gestures_from(lines)) == []


def test_selection_expires():
    lines = touch_lines(10.0, *Bot._get_card_centre(1)) + touch_lines(
        20.0, *Bot._get_tile_centre(5, 7)
    )
    assert gestures_to_plays(gestures_from(lines)) == []
