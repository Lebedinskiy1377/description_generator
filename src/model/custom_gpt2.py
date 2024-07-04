import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(
        variance + eps
    )  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]
    return x


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def attention(
        q, k, v, mask
):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]
    qkv_heads = list(
        map(lambda x: np.split(x, n_head, axis=-1), qkv)
    )  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]
    causal_mask = (1 - np.tri(x.shape[0])) * -1e10  # [n_seq, n_seq]
    out_heads = [
        attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)
    ]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x


def transformer_block(
        x, mlp, attn, ln_1, ln_2, n_head
):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention

    x_norm = layer_norm(x, **ln_1)

    x = mha(x_norm, **attn, n_head=n_head) + x  # [n_seq, n_embd] -> [n_seq, n_embd]

    x_pred = x.copy()

    # feed forward network

    x = layer_norm(x, **ln_2)

    x = ffn(x, **mlp) + x_pred  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(
            x, n_head=n_head, **block
        )  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    logits = x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]

    return logits[-1]
