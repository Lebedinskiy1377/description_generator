import numpy as np


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return np.dot(x, w) + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def ffn(x, c_fc, c_proj):
    frst_output = linear(x, c_fc.get("w"), c_fc.get("b"))
    nonlinear = gelu(frst_output)
    output = linear(nonlinear, c_proj.get("w"), c_proj.get("b"))
    return output