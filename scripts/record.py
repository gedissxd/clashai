"""Record yourself playing, to train the AI on (see docs/ai-training.md).

Start the OneJev screen detector and the emulator, then run:

    uv run python scripts/record.py

and play battles in the emulator as normal. Press Ctrl+C to stop. Each run
is saved to its own folder in recordings/.

To check that your taps are read correctly, first run:

    uv run python scripts/record.py --check-touch
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

from loguru import logger

from clashroyalebuildabot.ai.dataset import (
    RECORDING_VERSION,
    label_recording,
    load_recording,
)
from clashroyalebuildabot.ai.serialize import card_by_name, state_to_dict
from clashroyalebuildabot.ai.touch import (
    Calibration,
    GestureBuilder,
    card_slot_at,
    parse_event_line,
    parse_ranges,
    tile_at,
)
from clashroyalebuildabot.constants import ADB_PATH, APP_DIR, DECK_SIZE
from clashroyalebuildabot.gui.utils import load_config

RECORDINGS_DIR = os.path.join(APP_DIR, "recordings")


def resolve_deck(config):
    names = (config.get("ai") or {}).get("deck") or []
    if names:
        deck = [card_by_name(name) for name in names]
    else:
        # Fall back to the deck the rule-based bot uses in main.py.
        sys.path.insert(0, APP_DIR)
        from main import ACTIONS

        deck = [action.CARD for action in ACTIONS]
    if len(deck) != DECK_SIZE:
        raise SystemExit(f"Your deck needs {DECK_SIZE} cards, got {len(deck)}")
    return deck


def adb(serial, *args):
    return subprocess.run(
        [ADB_PATH, "-s", serial, *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def start_getevent(serial):
    # "-t -t" forces a terminal, so getevent sends each event straight away
    # instead of in batches.
    return subprocess.Popen(
        [ADB_PATH, "-s", serial, "shell", "-t", "-t", "getevent", "-lt"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )


def check_touch(serial, ranges, calibration):
    if not ranges:
        print("No touch screen found by 'getevent -lp'.")
        return
    print("Touch devices:", json.dumps(ranges))
    print(
        "Tap the four cards in your hand and a few places in the arena.\n"
        "Each tap is printed below. Press Ctrl+C to stop.\n"
    )
    builder = GestureBuilder(ranges, calibration)
    proc = start_getevent(serial)
    try:
        for line in proc.stdout:
            event = parse_event_line(line)
            if event is None:
                continue
            count = len(builder.gestures)
            builder.feed(event)
            for g in builder.gestures[count:]:
                slot = card_slot_at(g.x_down, g.y_down)
                tile = tile_at(g.x_down, g.y_down)
                where = (
                    f"hand card {slot + 1}"
                    if slot is not None
                    else f"arena tile {tile}"
                    if tile is not None
                    else "outside the arena and hand"
                )
                kind = "tap" if g.is_tap else "drag"
                print(
                    f"{kind} at ({g.x_down:.0f}, {g.y_down:.0f}) "
                    f"of 720x1280 -> {where}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()


class TouchWriter(threading.Thread):
    def __init__(self, proc, path):
        super().__init__(daemon=True)
        self.proc = proc
        self.file = open(path, "w", encoding="utf-8")
        self.count = 0

    def run(self):
        for line in self.proc.stdout:
            line = line.rstrip("\r\n")
            if not line.startswith("["):
                continue
            row = {"t": time.monotonic(), "line": line}
            self.file.write(json.dumps(row) + "\n")
            self.count += 1
            if "SYN_REPORT" in line:
                self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()


def record(emulator, detector, deck, ranges, calibration, save_frames):
    name = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(RECORDINGS_DIR, name)
    os.makedirs(out_dir)
    frames_dir = os.path.join(out_dir, "frames")
    if save_frames:
        os.makedirs(frames_dir)

    meta = {
        "version": RECORDING_VERSION,
        "deck": [card.name for card in deck],
        "touch_ranges": ranges,
        "calibration": vars(calibration),
        "screen_size": [emulator.width, emulator.height],
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    proc = start_getevent(emulator.device_serial)
    touches = TouchWriter(proc, os.path.join(out_dir, "touch.jsonl"))
    touches.start()

    print(f"Recording to {out_dir}. Play your battles, Ctrl+C to stop.")
    n_states = 0
    last_print = 0.0
    last_frame_save = 0.0
    with open(
        os.path.join(out_dir, "states.jsonl"), "w", encoding="utf-8"
    ) as f:
        try:
            while True:
                screenshot = emulator.take_screenshot()
                t = time.monotonic()
                if screenshot is None:
                    break
                state = detector.run(screenshot)
                if state is None:
                    continue
                f.write(json.dumps({"t": t, "state": state_to_dict(state)}))
                f.write("\n")
                n_states += 1
                if save_frames and t - last_frame_save >= 0.5:
                    screenshot.save(
                        os.path.join(frames_dir, f"{t:.3f}.jpg"), quality=85
                    )
                    last_frame_save = t
                if t - last_print >= 5:
                    f.flush()
                    print(
                        f"screen={state.screen.name:<20} "
                        f"elixir={state.numbers.elixir.number:<4} "
                        f"states={n_states} touch_events={touches.count}"
                    )
                    last_print = t
        except KeyboardInterrupt:
            pass
        finally:
            proc.terminate()
            touches.join(timeout=2)
            touches.close()
    return out_dir


def summarise(out_dir):
    from collections import Counter

    rec = load_recording(out_dir)
    stats = Counter()
    for _ in label_recording(rec, rec.deck, stats):
        pass
    print("\nRecording summary:")
    print(f"  states saved:          {len(rec.states)}")
    print(f"  card plays from taps:  {stats['taps_detected']}")
    print(f"  plays used for training: {stats['plays']}")
    for key, value in sorted(stats.items()):
        if key.startswith("skipped"):
            print(f"  {key}: {value}")
    if stats["taps_detected"] and not stats["plays"]:
        print(
            "\nNo plays could be used. Run with --check-touch and see "
            "docs/ai-training.md#check-your-taps."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check-touch",
        action="store_true",
        help="Only print where your taps land, to check the touch setup.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save two screenshots a second (about 30 MB per battle).",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    config = load_config()
    calibration = Calibration.from_config(
        (config.get("ai") or {}).get("touch")
    )

    # Imported here so --help works without the bot's heavy dependencies.
    from clashroyalebuildabot.detectors.detector import Detector
    from clashroyalebuildabot.emulator.emulator import Emulator

    emulator = Emulator(device_serial=config["adb"]["device_serial"])
    try:
        ranges = parse_ranges(
            adb(emulator.device_serial, "shell", "getevent", "-lp")
        )
        if args.check_touch:
            check_touch(emulator.device_serial, ranges, calibration)
            return

        deck = resolve_deck(config)
        print("Deck:", ", ".join(card.name for card in deck))
        detector = Detector(
            cards=deck, screen_config=config["screen_detector"]
        )
        try:
            out_dir = record(
                emulator, detector, deck, ranges, calibration, args.save_frames
            )
        finally:
            detector.close()
    finally:
        emulator.close()
    summarise(out_dir)


if __name__ == "__main__":
    main()
