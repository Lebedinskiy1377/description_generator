import numpy as np


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def attention(q, k, v):  # [n_q, d_k], [n_k, d_k], [n_k, d_v] -> [n_q, d_v]
    qkT = q @ k.T
    dk = np.sqrt(q.shape[-1])
    weights = softmax(qkT / dk)
    attention_vectors = weights @ v
    return attention_vectors


def self_attention(x, c_attn, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # QKV projections
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # Split into queries, keys, values
    q, k, v = np.split(x, 3, axis=1)

    # Perform self-attention mechanism
    x = attention(q, k, v)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # Output projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] @ [n_emb, n_embd] = [n_seq, n_embd]

    return x