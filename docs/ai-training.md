## Wiki: Train an AI on Your Own Games

Instead of the hand-written rules, the bot can play with a small neural
network that learned from watching you play. You record your battles, train
the model, then turn it on.

```
record your battles  ->  train  ->  the bot plays like you
 scripts/record.py      scripts/train.py   ai.enabled: True
```

The model copies you, so it plays about as well as you did in the recorded
battles, usually a bit worse. It only knows the deck you recorded with.

> [!TIP]
> Record in **Training Camp** or friendly battles. It's the same game for the
> model, and the bot never plays against people on the ladder while it's
> still learning.

---

### Before You Start

Set up the bot first ([Bot Installation](bot-installation.md)), including
the OneJev screen detector, and check that it runs. Recording uses the same
detectors to read the screen.

Pick a deck and keep it. Set it in `config.yaml`, by card name:

```yaml
ai:
  deck: [archers, goblin_barrel, baby_dragon, cannon, knight, minipekka, musketeer, witch]
```

Card names are in [`clashroyalebuildabot/namespaces/cards.py`](../clashroyalebuildabot/namespaces/cards.py).
If you leave `deck` empty, the deck from `main.py` is used. Unlike the
rule-based bot, any card the detectors know works, not just the cards
with an action.

---

### Check Your Taps

The recorder reads your taps from the emulator with `adb shell getevent`.
Check that it gets the positions right before recording lots of games. Start
a battle (Training Camp is fine), then run:

```bash
uv run python scripts/record.py --check-touch
```

Tap each of the four cards in your hand, left to right, then a few places in
the arena. You should see:

```
tap at (222, 1140) of 720x1280 -> hand card 1
tap at (358, 1140) of 720x1280 -> hand card 2
...
tap at (360, 700) of 720x1280 -> arena tile (9, 11)
```

If the hand cards come out wrong, your emulator's touch axes don't match the
screen. Fix them in `config.yaml` and check again:

| What you see | Setting |
| :--- | :--- |
| Tapping the bottom-left gives a top-left position | `flip_y: True` |
| Tapping the left side gives a right-side position | `flip_x: True` |
| Moving your finger right changes the second number | `swap_xy: True` |

```yaml
ai:
  touch:
    swap_xy: False
    flip_x: False
    flip_y: False
```

If nothing is printed at all, `getevent` can't see the emulator's touches.
In BlueStacks, check that ADB is turned on, then try another emulator
instance, and ask on Discord if it still doesn't work.

---

### Record Your Battles

With the screen detector and the emulator running, start the recorder and
play normally. Tap or drag cards, both work:

```bash
uv run python scripts/record.py
```

Every few seconds it prints what it sees. Play as many battles as you like,
then press `Ctrl+C`. Each run is saved to its own folder in `recordings/`
and ends with a summary:

```
Recording summary:
  states saved:          2140
  card plays from taps:  41
  plays used for training: 38
  skipped_not_confirmed: 3
```

A play is skipped when the card didn't leave your hand afterwards (a tap the
game didn't accept), or when the screen detector didn't think you were in a
battle. A few skips are normal. If most plays are skipped, check your taps
again.

Add `--save-frames` to also keep screenshots (about 30 MB per battle). You
don't need them to train, but they let you train again later with better
detectors without replaying.

**How many battles?** About 50 battles give a model that does something
sensible. 200 or more is much better. Play well and play consistently: the
model learns your habits, good and bad.

---

### Train

Install PyTorch once:

```bash
uv sync --extra cpu --extra train
```

Then train on everything in `recordings/`:

```bash
uv run python scripts/train.py
```

It prints a line per epoch:

```
epoch  12  train_loss 1.204  val_loss 1.311  play/wait 81%  card 63%  tile±1 55%
```

- **play/wait**: how often it chose to play or wait when you did.
- **card**: when you played, how often it picked the same card.
- **tile±1**: when you played, how often it put the card within one tile of
  where you did.

Training runs on the CPU unless PyTorch can use your NVIDIA GPU. On a
CPU, expect very roughly an hour for 50 battles and a few hours for 200.
Leave it running, or pass `--epochs 10` for a quicker, slightly weaker
model. The best epoch is kept, so stopping early with fewer epochs is safe.

> [!NOTE]
> Recent PyTorch builds for NVIDIA GPUs no longer support older cards such as
> the GTX 10 series. If `train.py` says `device=cpu` even though you have an
> NVIDIA GPU, that's why. Training on the CPU gives the same model, only
> slower.

The best epoch is saved to `ai_models/policy.onnx`, with `policy.json`
next to it. Training again replaces them, so copy them somewhere first if you
want to keep an old model. `--out` saves somewhere else.

---

### Play

Turn the model on in `config.yaml`:

```yaml
ai:
  enabled: True
```

Start the bot as usual (`uv run python main.py`) and make sure your deck in
the game is the one you recorded with. The log shows each move and how sure
the model was.

To tune it:

- **It waits too long:** lower `ai.play_threshold` (for example to `0.35`),
  or train with `--wait-weight 0.5`.
- **It spams cards:** raise `ai.play_threshold` (for example to `0.65`).
- **It plays in odd places:** record more battles. Placement needs the most
  data.

Set `ai.enabled` back to `False` to use the hand-written rules again.

---

### How It Works

Every screenshot becomes the [bot state](bot-state.md): units and their
tiles, elixir, tower HP and your hand. The recorder saves those states and
your raw touches. `scripts/train.py` matches each card you played to the
state just before you picked it, and labels every other in-battle moment
"wait".

The network ([`clashroyalebuildabot/ai/model.py`](../clashroyalebuildabot/ai/model.py))
reads the arena as an 18x32 grid of tiles, plus your hand, elixir and the
count of each unit. It answers two questions: which card to play, or wait,
and where each card should go. Training mirrors battles left-to-right, so
every recorded battle counts twice.

The bot only ever picks cards you can afford and tiles where they can be
placed. The trained model is exported to ONNX, so playing doesn't need
PyTorch.

**Next steps** once this works: give more weight to battles you won, or
fine-tune the model by letting it play and learn from wins and losses.
