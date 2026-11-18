import pytest
from test_ai_features import DECK
from test_bot_lifecycle import (
    ACTIONS,
    FakeDetector,
    FakeEmulator,
    FakeVisualizer,
)

import clashroyalebuildabot.bot.bot as bot_module
from clashroyalebuildabot.ai.policy import Decision
from clashroyalebuildabot.utils.error_handling import WikifiedError


class FakePolicy:
    deck = DECK

    def __init__(self, decision):
        self.decision = decision

    def decide(self, state):
        return self.decision


class ClickingEmulator(FakeEmulator):
    def __init__(self, device_serial):
        super().__init__(device_serial)
        self.clicks = []

    def click(self, x, y):
        self.clicks.append((x, y))


def make_config(enabled=True, model_path="ai_models/policy.onnx"):
    return {
        "bot": {"auto_start_game": False, "load_deck": False},
        "adb": {"device_serial": "test-device"},
        "visuals": {},
        "ingame": {"play_action": 0},
        "ai": {"enabled": enabled, "model_path": model_path},
    }


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(bot_module, "Emulator", ClickingEmulator)
    monkeypatch.setattr(bot_module, "Detector", FakeDetector)
    monkeypatch.setattr(bot_module, "Visualizer", FakeVisualizer)


def test_learned_decision_is_played(monkeypatch, state_factory):
    decision = Decision(
        slot=2, tile_x=9, tile_y=6, card_name="x", confidence=0.9
    )
    monkeypatch.setattr(
        bot_module, "_load_policy", lambda config: FakePolicy(decision)
    )
    bot = bot_module.Bot(ACTIONS, make_config())

    bot._handle_game_step(state_factory())

    assert bot.emulator.clicks == [
        bot._get_card_centre(2),
        bot._get_tile_centre(9, 6),
    ]


def test_learned_wait_does_not_click(monkeypatch, state_factory):
    monkeypatch.setattr(
        bot_module, "_load_policy", lambda config: FakePolicy(None)
    )
    bot = bot_module.Bot(ACTIONS, make_config())

    bot._handle_game_step(state_factory())

    assert bot.emulator.clicks == []


def test_rules_are_used_when_ai_is_off():
    bot = bot_module.Bot(ACTIONS, make_config(enabled=False))
    assert bot.policy is None


def test_missing_model_raises_e011(tmp_path):
    missing = str(tmp_path / "nope.onnx")
    with pytest.raises(WikifiedError, match="E011"):
        bot_module.Bot(ACTIONS, make_config(model_path=missing))
