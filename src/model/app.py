import logging
from typing import List

import numpy as np
import streamlit as st
from custom_gpt2 import gpt2
from util import load_encoder_hparams_and_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit")

# PARAMETERS
MODEL_SIZE = "124M"
MODELS_DIR = "models"
N_TOKENS = 50


def product_examples(filename: str = "product_examples.txt") -> str:
    """Load product examples from the given file."""
    with open(filename) as f:
        return f.read()


def seo_prompt(examples: str, product_title: str) -> str:
    """Construct an SEO optimization input from examples and new product title."""
    prompt = f"{examples}\n\nTitle: {product_title}.\nDescription:"
    return prompt


def generate(
    input_ids: List[int],
    n_tokens: int,
    n_head: int,
    top_k: int = 100,
    top_p: float = 0.95,
    temperature: float = 1.0,
    weights: dict = None,
    random_state: int = 0,
    stop_tokens: List[int] = None,
) -> List[int]:
    """
    Generate text using GPT-2 model.

    Parameters:
    - input_ids: Initial token IDs for the generation.
    - ... [other parameters]
    - stop_tokens: Token IDs that signal the end of generation.

    Returns:
    - A list of generated token IDs.
    """

    np.random.seed(random_state)
    output_ids = []

    # Initialize progress bar in Streamlit
    progress = st.progress(0)

    for i in range(n_tokens):
        logits = gpt2(input_ids + output_ids, n_head=n_head, **weights)
        logits /= temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))

        # Apply top-k and top-p filtering
        indices = np.argsort(probs)[-top_k:]
        probs = probs[indices] / probs[indices].sum()

        # Select next token probabilistically
        next_id = int(np.random.choice(indices, p=probs))
        output_ids.append(next_id)

        # Update the progress bar
        progress.progress((i + 1) / n_tokens)

        # Stop if end or stop token is generated
        if stop_tokens and next_id in stop_tokens:
            break

    progress.empty()

    return output_ids


# Load model components
encoder, hparams, weights = load_encoder_hparams_and_weights(
    MODEL_SIZE, MODELS_DIR
)

# Initialize Streamlit app
st.title(f"AI-SEO [GPT-2, {MODEL_SIZE}]")

# Load product examples from the file
examples = product_examples("products.txt")

# Get new product title from user
new_product_title = st.text_input("New Product Title:")

if st.button("Generate Description"):
    # Construct SEO optimization input
    input_str = seo_prompt(examples, new_product_title)
    logger.info(f"Input: {input_str}")

    input_ids = encoder.encode(input_str)
    assert len(input_ids) + N_TOKENS < hparams["n_ctx"]

    stop_tokens = [encoder.encode(token)[0] for token in ["\n"]]

    # Generate a response
    output_ids = generate(
        input_ids=input_ids,
        n_tokens=N_TOKENS,
        n_head=hparams["n_head"],
        top_k=50,
        top_p=0.8,
        temperature=0.75,
        weights=weights,
        stop_tokens=[50256] + stop_tokens,
    )

    response = encoder.decode(output_ids).strip()
    logger.info(f"Output: {response}")

    st.subheader("Generated Description:")
    st.write(f"{response}")
