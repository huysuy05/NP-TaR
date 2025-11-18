import random
import string

def generate_low_signal(original_texts, strategy='mask'):
    """
    Generate low-signal variants of texts using different strategies.
    
    Args:
        original_texts: List of original text strings
        strategy: One of 'mask', 'stopwords', 'random', 'length', 'punctuation', 'mixed'
    
    Returns:
        List of low-signal text variants
    """
    if strategy == 'mask':
        # Constant mask token
        return ["[MASK]" for _ in original_texts]
    
    elif strategy == 'stopwords':
        # Only common stopwords (no semantic content)
        stopwords = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"]
        return [" ".join(random.choices(stopwords, k=random.randint(3, 10))) for _ in original_texts]
    
    
    elif strategy == 'length':
        # Just preserve length with 'x' characters
        return ["x" * len(text) for text in original_texts]
    
    elif strategy == 'punctuation':
        # Only punctuation marks
        punct = [".", ",", "!", "?", ";", ":", "-"]
        return [" ".join(random.choices(punct, k=random.randint(5, 15))) for _ in original_texts]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
