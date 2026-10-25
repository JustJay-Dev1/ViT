"""Print a concise summary from the real history.json created by train.py.

Run from your project root, e.g.:
    python summarize_training_for_ppt.py --history checkpoints/history.json

This reads saved training history; it does NOT re-train the model.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="checkpoints/history.json", help="history.json written by train.py")
    args = parser.parse_args()
    path = Path(args.history).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"ERROR: History file not found: {path}\nPass the path to the history.json from your completed PC run.")

    with path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list) or not history:
        raise SystemExit("ERROR: history.json has no epoch records.")
    best = max(history, key=lambda row: row.get("val_acc", float("-inf")))
    last = history[-1]

    print("=" * 60)
    print("COMPLETED BASELINE TRAINING — SAVED HISTORY")
    print("=" * 60)
    print(f"History file          : {path}")
    print(f"Epoch records         : {len(history)}")
    print(f"Best validation epoch : {best.get('epoch', 'unknown')}")
    print(f"Best validation acc   : {best['val_acc'] * 100:.4f}%")
    print(f"Best validation loss  : {best['val_loss']:.6f}")
    print(f"Last recorded epoch   : {last.get('epoch', 'unknown')}")
    print(f"Last train accuracy   : {last['train_acc'] * 100:.4f}%")
    print(f"Last validation acc   : {last['val_acc'] * 100:.4f}%")
    print(f"Last train loss       : {last['train_loss']:.6f}")
    print(f"Last validation loss  : {last['val_loss']:.6f}")
    print("\nThese values are read from the saved history; no training was run.")


if __name__ == "__main__":
    main()
