# import torch
import numpy as np
import os
import tiktoken

enc = tiktoken.get_encoding("gpt2")

V = enc.n_vocab
d = 128
n_h = 4
d_h = d//n_h # 32
b = 10000

# With this function, the size of matrix E is set into stone, which is bad
def E_Setup():
    if os.path.exists("embeddings.npy"):
        E = np.load("embeddings.npy")
    else:
        E = np.random.rand(d, V)
        np.save("embeddings.npy", E)
    return E

E = E_Setup()

def gamma1_Setup():
    if os.path.exists("gamma.npy"):
        gamma = np.load("gamma.npy")
    else:
        gamma = np.ones((d, 1))
        np.save("gamma.npy", gamma)
    return gamma

def gamma2_Setup():
    if os.path.exists("gamma2.npy"):
        gamma = np.load("gamma2.npy")
    else:
        gamma = np.ones((d, 1))
        np.save("gamma2.npy", gamma)
    return gamma

def gammaFin_Setup():
    if os.path.exists("gammaFin.npy"):
        gamma = np.load("gammaFin.npy")
    else:
        gamma = np.ones((d, 1))
        np.save("gammaFin.npy", gamma)
    return gamma

y_1 = gamma1_Setup()
y_2 = gamma2_Setup()
y_fin = gammaFin_Setup()

W_Q = np.random.randn(n_h, d_h, d) # To be replaced with persistant matricies
W_K = np.random.randn(n_h, d_h, d)
W_V = np.random.randn(n_h, d_h, d)

W_O = np.random.randn(d, d)

d_ff = 4 * d  # 512

W_gate = np.random.randn(d_ff, d)
W_up = np.random.randn(d_ff, d)
W_down = np.random.randn(d, d_ff)
#W_out = np.random.randn(V, d) * 0.02
W_out = E.T

def main():
    while True:
        text = input(">> ")

        if text == "quit":
            break

        # 1. Embed the words
        T = enc.encode(text)
        print(T)
        X = embed(T)
        H = X
        print(X.shape) # h x h What the heck is h? Idk what I was thinking here
        print(X[0][0])

        # 2. Normalize
        U = RMSNorm(H, y_1) # h x n

        # 3. Project to queries, keys, and values AND 4. Position encoding with RoPE
        Q = RoPE(W_Q @ U) # h_q x n 
        K = RoPE(W_K @ U)
        Vals = W_V @ U

        # 5. Scaled dot product attention and casual masks AND 6. Softmax AND 7. Weighted average of value vectors
        Z = weighted_average(softmax(causal_mask(scale_product_attention(Q, K))), Vals) # n_h, n x n (I combined everything because why not)

        # 8. Stacking the heads into a mega-vector
        M = np.vstack(Z) # Can also use Z = Z.reshape(d, len(T)) but I dont understand that one
        H_p = H + W_O@M

        # 9. Normalize + Feed Forward
        W = RMSNorm(H_p, y_2)
        F = feed_forward(W)
        H_out = H_p + F

        # 10. Next token generation
        H_hat = RMSNorm(H_out, y_fin)
        L = W_out @ H_hat

        # I got too lazy and GPT completed this last bit of code:
        logits = L[:, -1]              # (V,)
        # Softmax
        exp_L = np.exp(logits - np.max(logits))
        V_l = exp_L / np.sum(exp_L)     # (V,)

        next_token = np.argmax(V_l)
        print(enc.decode([int(next_token)]))

def embed(tok_ids):
    E_t = np.zeros((V, len(tok_ids)))
    for i in range(len(tok_ids)):
        E_t[tok_ids[i], i] = 1
    return E@E_t

def RMS(H):
    return np.sqrt(np.mean(H**2, axis=0, keepdims=True) + 1e-6)

def RMSNorm(H, y):
    return y * H/RMS(H)

def RoPE(M):
    R = M.copy()

    for i in range(n_h):
        for m in range(M.shape[2]):
            for k in range(d_h // 2):
                angle = m * b**(-2*k/d_h)

                R[i, 2*k, m] = M[i, 2*k, m] * np.cos(angle) - M[i, 2*k+1, m] * np.sin(angle)
                R[i, 2*k+1, m] = M[i, 2*k, m] * np.sin(angle) + M[i, 2*k+1, m] * np.cos(angle)

    return R

def scale_product_attention(Q, K):
    return Q.swapaxes(1, 2) @ K / np.sqrt(d_h) # I'm not good enough at keeping track of dims

def causal_mask(S): # is S square? idk. I'm pretty confused here
    R = S.copy()
    for h in range(n_h):
        for i in range(R.shape[1]):
            for j in range(i + 1, R.shape[2]):
                R[h, i, j] = -np.inf

    return R

def softmax(S): # I can subtract maximum if I want to? also is this a row = tok or col = tok matrix and idk which one im supposed to use
    R = S.copy()
    for h in range(n_h):
        for i in range(R.shape[1]):
            max = np.max(R[h, i, :])
            total = 0
            for j in range(R.shape[2]):
                total += np.exp(R[h][i][j]-max)
            for k in range(R.shape[2]):
                R[h][i][k] = np.exp(R[h][i][k]-max)/total
    return R

def weighted_average(A, V):
    return V @ A.transpose(0, 2, 1)

def SiLU(X):
    return X / (1 + np.exp(-X))

def feed_forward(W):
    return W_down @ (SiLU(W_gate @ W) * (W_up @ W))

main()