import argparse
import codecs
import math
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np  # Used only for importing weights from the original script.
import tiktoken
import torch
import torch.nn.functional as F

enc = tiktoken.get_encoding("gpt2")

V = enc.n_vocab
d = 768        # Can also try 128
n_h = 12       # Use 4 when d = 128.
d_h = d // n_h
d_ff = 4 * d
b = 10000

if d % n_h != 0 or d_h % 2 != 0:
    raise ValueError("Heads must divide d evenly, and RoPE needs an even d_h.")

# Change the dimensions to start another model. Change them back to resume this one.
checkpoint = Path(f"weights/{enc.name}_V{V}_d{d}_heads{n_h}_ff{d_ff}_b{b}.pt")
saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint.exists() else None
parameters = {}


def matrix(name, shape, *, ones=False, old_file=None):
    if saved is not None:
        tensor = saved[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{checkpoint}: expected {name} to have shape {shape}, got {tuple(tensor.shape)}")
    else:
        tensor = None
        if old_file is not None and Path(old_file).is_file():
            old = np.load(old_file, mmap_mode="r", allow_pickle=False)
            if old.shape == shape:
                tensor = torch.tensor(old, dtype=torch.float32)
        if tensor is None:
            tensor = torch.ones(shape) if ones else torch.randn(shape) * 0.02

    parameters[name] = tensor.requires_grad_()
    return parameters[name]


def save_weights():
    checkpoint.parent.mkdir(exist_ok=True)
    # Replace the previous checkpoint only after the entire new one has been written.
    with NamedTemporaryFile(dir=checkpoint.parent, delete=False) as temp:
        temporary_path = Path(temp.name)
    try:
        torch.save({name: p.detach() for name, p in parameters.items()}, temporary_path)
        temporary_path.replace(checkpoint)
    finally:
        temporary_path.unlink(missing_ok=True)


E = matrix("E", (d, V), old_file="embeddings.npy")
y_1 = matrix("y_1", (d, 1), ones=True, old_file="gamma.npy")
y_2 = matrix("y_2", (d, 1), ones=True, old_file="gamma2.npy")
y_fin = matrix("y_fin", (d, 1), ones=True, old_file="gammaFin.npy")

W_Q = matrix("W_Q", (n_h, d_h, d))
W_K = matrix("W_K", (n_h, d_h, d))
W_V = matrix("W_V", (n_h, d_h, d))
W_O = matrix("W_O", (d, d))

W_gate = matrix("W_gate", (d_ff, d))
W_up = matrix("W_up", (d_ff, d))
W_down = matrix("W_down", (d, d_ff))

if saved is None:
    save_weights()


@torch.no_grad()
def generate_chunks(prompt_tokens, temperature=0.5):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and non-negative.")
    tokens = list(prompt_tokens)
    if not tokens:
        return
    limit = 200
    keys = E.new_empty((n_h, d_h, len(tokens) + limit - 1))
    values = torch.empty_like(keys)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    start = 0

    for _ in range(limit):
        logits = cached_next_logits(tokens, start, keys, values)
        if temperature == 0:
            next_token = logits.argmax().item()
        else:
            scores = logits.flatten()
            probabilities = torch.softmax((scores - scores.max()) / temperature, dim=0)
            next_token = torch.multinomial(probabilities, 1).item()
        if next_token == enc.eot_token:
            break
        chunk = decoder.decode(enc.decode_single_token_bytes(next_token))
        if chunk:
            yield chunk
        start += len(tokens)
        tokens = [next_token]
    tail = decoder.decode(b"", final=True)
    if tail:
        yield tail


def generate(prompt_tokens, temperature=0.8):
    return "".join(generate_chunks(prompt_tokens, temperature))


def cached_next_logits(tokens, start, keys, values):
    # Prefill the prompt, then append one token on each decoding step.
    end = start + len(tokens)
    H = embed(tokens)
    U = RMSNorm(H, y_1)
    keys[:, :, start:end] = RoPE(W_K @ U, start)
    values[:, :, start:end] = W_V @ U
    # The newest query can attend to every filled slot.
    Q = RoPE(W_Q @ U[:, -1:], end - 1)
    A = softmax(scale_product_attention(Q, keys[:, :, :end]))
    Z = weighted_average(A, values[:, :, :end])
    return output_logits(H[:, -1:], Z)


def main():
    from train_madao import place_weights

    parser = argparse.ArgumentParser(description="Chat with MADAO.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 chooses the most likely token; default: 0.8)")
    parser.add_argument("--system-prompt", default="You are an AI agent named MADAO, complete this text using the prior context.",
                        help="Instruction prepended to each generation prompt; use an empty string to disable")
    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("Temperature must be finite and non-negative.")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    place_weights(sys.modules[__name__], device)
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU", flush=True)
    optimizer = torch.optim.SGD(parameters.values(), lr=0.01)
    while True:
        text = input(">> ")
        if text == "quit":
            break
        if text.startswith("/train "):
            tokens = enc.encode(text[len("/train "):])
            if len(tokens) < 2:
                print("Training needs at least two tokens.")
                continue
            loss = train_step(tokens, optimizer)
            save_weights()
            print(f"loss = {loss:.4f}")
            continue

        if text:
            prompt = f"{args.system_prompt}\n\n{text}" if args.system_prompt else text
            tokens = enc.encode(prompt)
            for chunk in generate_chunks(tokens, args.temperature):
                print(chunk, end="", flush=True)
            print(flush=True)


def forward(T):
    # 1. Embed the tokens. Each column is one token.
    X = embed(T)                         # (d, n), where n = len(T)
    H = X

    # 2. Normalize
    U = RMSNorm(H, y_1)

    # 3. Project to queries, keys, and values. 4. Rotate queries and keys.
    Q = RoPE(W_Q @ U)                    # (n_h, d_h, n)
    K = RoPE(W_K @ U)
    Vals = W_V @ U

    # 5–7. Score, mask future tokens, normalize each row, then average values.
    A = softmax(causal_mask(scale_product_attention(Q, K)))  # (n_h, n, n)
    Z = weighted_average(A, Vals)        # (n_h, d_h, n)
    return output_logits(H, Z)


def output_logits(H, Z):
    # 8. Stack the heads vertically, then add the residual.
    M = torch.vstack(Z.unbind(0))        # (d, n)
    H_p = H + W_O @ M

    # 9. Normalize + feed forward
    W = RMSNorm(H_p, y_2)
    H_out = H_p + feed_forward(W)

    # 10. One column of next-token logits for every input position.
    H_hat = RMSNorm(H_out, y_fin)
    W_out = E.T                          # Shares E's weights and its gradients.
    return W_out @ H_hat                 # (V, n)


def embed(tok_ids):
    # Multiplying E by a one-hot vector selects exactly this column of E.
    return E[:, tok_ids]

def RMS(H):
    return torch.sqrt(torch.mean(H**2, dim=0, keepdim=True) + 1e-6)

def RMSNorm(H, y):
    return y * H / RMS(H)

def RoPE(M, start=0):
    # Same angle as m * b**(-2*k/d_h), for every pair k and position m.
    k = torch.arange(d_h // 2, device=M.device, dtype=M.dtype)
    m = torch.arange(start, start + M.shape[2], device=M.device, dtype=M.dtype)
    angle = (b ** (-2 * k / d_h))[:, None] * m
    cos, sin = torch.cos(angle), torch.sin(angle)

    even = M[:, 0::2, :]
    odd = M[:, 1::2, :]
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    # Place each rotated even coordinate directly before its odd partner.
    return torch.stack((rotated_even, rotated_odd), dim=2).flatten(1, 2)

def scale_product_attention(Q, K):
    return Q.transpose(1, 2) @ K / math.sqrt(d_h)

def causal_mask(S):
    # S[h, i, j] scores how much query i attends to key j. Hide j > i.
    n = S.shape[-1]
    future = torch.ones((n, n), dtype=torch.bool, device=S.device).triu(1)
    return S.masked_fill(future, -torch.inf)

def softmax(S):
    return torch.softmax(S, dim=-1)      # Each query's row sums to 1.

def weighted_average(A, V):
    return V @ A.transpose(1, 2)

def SiLU(X):
    return X * torch.sigmoid(X)

def feed_forward(W):
    return W_down @ (SiLU(W_gate @ W) * (W_up @ W))

def train_step(tokens, optimizer):
    # With [a, b, c], predict [b, c] from the input [a, b].
    optimizer.zero_grad()
    L = forward(tokens[:-1])             # (V, n-1)
    targets = torch.tensor(tokens[1:], dtype=torch.long, device=L.device)
    loss = F.cross_entropy(L.T, targets)  # Cross-entropy expects raw logits.
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == "__main__":
    main()
