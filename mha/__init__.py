"""
Multi-Head Attention (MHA) Implementation
Standard transformer attention mechanism from "Attention Is All You Need"

Example usage:
    from mha import Transformer

    model = Transformer(
        vocab_size=50257,
        d_model=512,
        num_heads=8
    )
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main classes for easy access
from .transformer import Transformer, TransformerEncoder, TransformerDecoder, EncoderLayer, DecoderLayer
from .attention import (
    MultiHeadAttention,
    ScaledDotProductAttention,
    create_causal_mask,
    create_padding_mask,
    create_combined_mask
)
from .layers import LayerNorm, FeedForward, ResidualConnection
from .positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    PositionalEncodingFactory
)

# Define what gets imported with "from mha import *"
__all__ = [
    # Version info
    '__version__',
    '__author__',

    # Main transformer
    'Transformer',
    'TransformerEncoder',
    'TransformerDecoder',
    'EncoderLayer',
    'DecoderLayer',

    # Attention mechanisms
    'MultiHeadAttention',
    'ScaledDotProductAttention',
    'create_causal_mask',
    'create_padding_mask',
    'create_combined_mask',

    # Layers
    'LayerNorm',
    'FeedForward',
    'ResidualConnection',

    # Positional encodings
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEncoding',
    'PositionalEncodingFactory',
]
