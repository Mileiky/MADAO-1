"""Train a MADAO model on a UTF-8 text file.
Run: python -B train_madao.py corpus.txt
MADAO saves trained weights after a successful run. Metrics and loss graphs go to the terminal.
"""
import argparse
import codecs
import importlib.util
import math
import mmap
import random
import sys
from importlib import import_module
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F


def place_weights(model, device):
    # forward() reads names like E and W_Q. The optimizer reads the dictionary.
    # Give both the same trainable tensor before constructing the optimizer.
    for name, weight in model.parameters.items():
        if weight.device.type != device:
            moved = weight.detach().to(device).requires_grad_()
            model.parameters[name] = moved
            setattr(model, name, moved)


def text_window(data, start, split_start, split_end, context, encoding):
    """Tokenize a bounded UTF-8 excerpt and return a full prediction window."""
    size = max(256, (context + 1) * 8)
    position = start
    while True:
        end = min(position + size, split_end)
        while position < end and data[position] & 0xC0 == 0x80:
            position += 1
        decoder = codecs.getincrementaldecoder("utf-8")()
        text = decoder.decode(data[position:end], final=end == split_end)
        tokens = encoding.encode_ordinary(text)
        if len(tokens) > context:
            return tokens[:context + 1]
        if end < split_end:
            size *= 2
        elif position > split_start:
            position = max(split_start, position - size)
            size *= 2
        else:
            raise ValueError("Corpus split has too few tokens for the requested context.")


def fixed_windows(data, start, end, context, count, encoding):
    positions = [start + i * (end - start - 1) // max(1, count - 1) for i in range(count)]
    return [text_window(data, position, start, end, context, encoding) for position in positions]


def load_model(name):
    path = Path(name if name.endswith(".py") else name + ".py")
    if path.is_file():
        module_name = "_madao_training_model"
        spec = importlib.util.spec_from_file_location(module_name, path)
        model = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = model
        spec.loader.exec_module(model)
        return model
    return import_module(name)


@torch.no_grad()
def evaluate(model, windows):
    total_loss = correct = total = 0
    for tokens in windows:
        logits = model.forward(tokens[:-1]).T  # (positions, vocabulary)
        targets = torch.tensor(tokens[1:], device=logits.device)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.numel()
    return total_loss / total, correct / total


def print_graph(train_history, val_history):
    """One character per evaluation, with the same loss scale for both lines."""
    train = train_history[-48:]
    val = val_history[-48:]
    low = min(min(train), min(val))
    high = max(max(train), max(val))
    bars = "▁▂▃▄▅▆▇█"

    def line(values):
        if high == low:
            return bars[0] * len(values)
        return "".join(bars[round((x - low) / (high - low) * 7)] for x in values)

    print(f"Loss from {low:.3f} to {high:.3f} | last {len(val)} evaluations, oldest first")
    print("  train  " + line(train))
    print("  valid  " + line(val), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--model", default="MADAO", help="Model filename, with or without .py")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-windows", type=int, default=8, help="Maximum windows per split")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.context, args.eval_every, args.eval_windows) < 1:
        parser.error("Steps, context, eval-every and eval-windows must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Supply a positive finite learning rate.")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment. Check nvidia-smi and your PyTorch build.")
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU", flush=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = load_model(args.model)
    corpus_file = args.corpus.open("rb")
    if args.corpus.stat().st_size == 0:
        parser.error("Corpus is empty.")
    data = mmap.mmap(corpus_file.fileno(), 0, access=mmap.ACCESS_READ)
    split = len(data) * 9 // 10
    while split < len(data) and data[split] & 0xC0 == 0x80:
        split += 1
    if split == len(data):
        parser.error("Corpus is too short for a validation split.")
    try:
        train_windows = fixed_windows(data, 0, split, args.context, args.eval_windows, model.enc)
        val_windows = fixed_windows(data, split, len(data), args.context, args.eval_windows, model.enc)
    except ValueError as error:
        parser.error(str(error))
    place_weights(model, device)
    optimizer = torch.optim.SGD(model.parameters.values(), lr=args.lr)

    print(f"{split:,} training bytes | {len(data) - split:,} validation bytes")
    print(f"Fixed evaluation targets: {len(train_windows) * args.context} training, "
          f"{len(val_windows) * args.context} validation")
    print(" step   tokens   train loss   val loss   val accuracy   val perplexity   train tok/s", flush=True)

    train_history, val_history = [], []
    train_seconds = 0.0
    last_report = 0
    for step in range(args.steps + 1):
        if step > 0:
            tick = perf_counter()
            start = rng.randrange(split)
            window = text_window(data, start, 0, split, args.context, model.enc)
            update_loss = model.train_step(window, optimizer)
            train_seconds += perf_counter() - tick
            if not math.isfinite(update_loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}.")

        if step % args.eval_every != 0 and step != args.steps:
            continue

        train_loss, _ = evaluate(model, train_windows)
        val_loss, val_acc = evaluate(model, val_windows)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise FloatingPointError(f"Nonfinite evaluation loss at step {step}.")
        train_history.append(train_loss)
        val_history.append(val_loss)
        perplexity = math.exp(val_loss) if val_loss < 709 else math.inf
        speed = f"{(step - last_report) * args.context / train_seconds:.0f}" if train_seconds else "-"
        print(f"{step:5d} {step * args.context:8d} {train_loss:12.3f} {val_loss:10.3f} "
              f"{val_acc:13.1%} {perplexity:16.1f} {speed:13}", flush=True)
        last_report = step
        train_seconds = 0.0

    save_weights = getattr(model, "save_weights", None)
    if save_weights is not None:
        save_weights()
        print("Saved trained weights.", flush=True)

    print_graph(train_history, val_history)
    data.close()
    corpus_file.close()


if __name__ == "__main__":
    main()



