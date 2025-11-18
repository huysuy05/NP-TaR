import random

def generate_noise(original_texts):
    """
    Produce shuffled versions of original texts.
    Keeps length but destroys coherent content.
    """
    noisy = []
    for tok in original_texts:
        tokens = tok.split()
        random.shuffle(tokens)
        # append back the shuffled text
        noisy.append(" ".join(tokens))

    return noisy

# Backward-compatible alias
generate_noise_inputs = generate_noise
