import numpy as np


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def softmax(x):
    # We subtract max(x) for numerical stability
    # https://jaykmody.com/blog/stable-softmax/
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def attention(
    q, k, v, mask
):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    qkT = q @ k.T
    dk = np.sqrt(q.shape[-1])
    weights = softmax((qkT / dk) + mask)
    attention_vectors = weights @ v
    return attention_vectors


def causal_self_attention(x, c_attn, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    q, k, v = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0])) * -1e10  # [n_seq, n_seq]

    # perform causal self attention and make projection
    x = attention(q, k, v, causal_mask)  # [n_seq, n_embd] -> [n_seq, n_embd]
    x = linear(x, **c_proj)  # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = [np.array_split(matr, n_head, axis=-1) for matr in qkv]  # [3, n_seq, n_embd] -> [n_head, n_seq, n_embd]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0])) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = np.array([attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)])

    # merge heads
    x = np.concatenate(out_heads, axis=-1)

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x
