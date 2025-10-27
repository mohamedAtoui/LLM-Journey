"""
Full Transformer Encoder-Decoder Architecture with Multi-Head Attention
Based on "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention, create_combined_mask, create_padding_mask
from .layers import LayerNorm, FeedForward, ResidualConnection
from .positional_encoding import PositionalEncodingFactory


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer

    Architecture:
        1. Multi-Head Self-Attention + Residual + LayerNorm
        2. Feed-Forward Network + Residual + LayerNorm

    Input shape:  (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Residual connections
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) - Optional padding mask

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.residual1(x, lambda x: self.self_attention(x, x, x, mask)[0])

        # Feed-forward with residual connection
        output = self.residual2(attn_output, self.feed_forward)

        return output


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer

    Architecture:
        1. Masked Multi-Head Self-Attention + Residual + LayerNorm
        2. Multi-Head Cross-Attention (to encoder) + Residual + LayerNorm
        3. Feed-Forward Network + Residual + LayerNorm

    Input shape:
        x (decoder input): (batch_size, seq_len_tgt, d_model)
        encoder_output: (batch_size, seq_len_src, d_model)
    Output shape: (batch_size, seq_len_tgt, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Multi-head cross-attention (decoder attends to encoder)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Residual connections
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch_size, seq_len_tgt, d_model) - Decoder input
            encoder_output: (batch_size, seq_len_src, d_model) - Encoder output
            src_mask: (batch_size, seq_len_src, seq_len_src) - Source padding mask
            tgt_mask: (batch_size, seq_len_tgt, seq_len_tgt) - Target causal + padding mask

        Returns:
            output: (batch_size, seq_len_tgt, d_model)
        """
        # Masked self-attention with residual connection
        self_attn_output = self.residual1(
            x, lambda x: self.self_attention(x, x, x, tgt_mask)[0]
        )

        # Cross-attention to encoder with residual connection
        # Query from decoder, Key and Value from encoder
        cross_attn_output = self.residual2(
            self_attn_output,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)[0]
        )

        # Feed-forward with residual connection
        output = self.residual3(cross_attn_output, self.feed_forward)

        return output


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder (stack of N encoder layers)

    Input shape:  (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) - Optional padding mask

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer normalization
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder (stack of N decoder layers)

    Input shape:
        x: (batch_size, seq_len_tgt, d_model)
        encoder_output: (batch_size, seq_len_src, d_model)
    Output shape: (batch_size, seq_len_tgt, d_model)
    """
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: (batch_size, seq_len_tgt, d_model) - Decoder input
            encoder_output: (batch_size, seq_len_src, d_model) - Encoder output
            src_mask: (batch_size, seq_len_src, seq_len_src) - Source padding mask
            tgt_mask: (batch_size, seq_len_tgt, seq_len_tgt) - Target causal + padding mask

        Returns:
            output: (batch_size, seq_len_tgt, d_model)
        """
        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Final layer normalization
        return self.norm(x)


class Transformer(nn.Module):
    """
    Full Transformer Encoder-Decoder Architecture with MHA

    Architecture:
        1. Input Embedding + Positional Encoding
        2. Encoder (N layers)
        3. Decoder (N layers)
        4. Output Linear + Softmax

    For language modeling on WikiText:
        - Source and target are the same (shifted by 1)
        - Use causal masking in decoder
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pe_type: str = "sinusoidal"
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embeddings
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encodings
        self.src_pos_encoding = PositionalEncodingFactory.create(
            pe_type, d_model, max_seq_length, dropout
        )
        self.tgt_pos_encoding = PositionalEncodingFactory.create(
            pe_type, d_model, max_seq_length, dropout
        )

        # Encoder and Decoder
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters with Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: (batch_size, seq_len_src) - Source token IDs
            tgt: (batch_size, seq_len_tgt) - Target token IDs
            src_mask: (batch_size, seq_len_src, seq_len_src) - Source mask
            tgt_mask: (batch_size, seq_len_tgt, seq_len_tgt) - Target mask

        Returns:
            output: (batch_size, seq_len_tgt, vocab_size) - Logits over vocabulary
        """
        # Embed and add positional encoding
        # src_embedded: (batch_size, seq_len_src, d_model)
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.src_pos_encoding(src_embedded)

        # tgt_embedded: (batch_size, seq_len_tgt, d_model)
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)

        # Encode source
        # encoder_output: (batch_size, seq_len_src, d_model)
        encoder_output = self.encoder(src_embedded, src_mask)

        # Decode target
        # decoder_output: (batch_size, seq_len_tgt, d_model)
        decoder_output = self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary
        # output: (batch_size, seq_len_tgt, vocab_size)
        output = self.output_projection(decoder_output)

        return output


import math


if __name__ == "__main__":
    # Unit tests for Transformer
    print("Testing Transformer Architecture...")

    batch_size = 2
    seq_len_src = 10
    seq_len_tgt = 8
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    # Test 1: EncoderLayer
    print("\n1. Testing EncoderLayer...")
    encoder_layer = EncoderLayer(d_model, num_heads, d_ff).to(device)
    x = torch.randn(batch_size, seq_len_src, d_model).to(device)
    output = encoder_layer(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "EncoderLayer output shape mismatch"

    # Test 2: DecoderLayer
    print("\n2. Testing DecoderLayer...")
    decoder_layer = DecoderLayer(d_model, num_heads, d_ff).to(device)
    tgt = torch.randn(batch_size, seq_len_tgt, d_model).to(device)
    encoder_output = torch.randn(batch_size, seq_len_src, d_model).to(device)
    output = decoder_layer(tgt, encoder_output)
    print(f"   Target shape: {tgt.shape}")
    print(f"   Encoder output shape: {encoder_output.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == tgt.shape, "DecoderLayer output shape mismatch"

    # Test 3: Full Transformer
    print("\n3. Testing Full Transformer...")
    transformer = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=512,
        dropout=0.1,
        pe_type="sinusoidal"
    ).to(device)

    src_tokens = torch.randint(0, vocab_size, (batch_size, seq_len_src)).to(device)
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, seq_len_tgt)).to(device)

    output = transformer(src_tokens, tgt_tokens)
    print(f"   Source tokens shape: {src_tokens.shape}")
    print(f"   Target tokens shape: {tgt_tokens.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len_tgt, vocab_size), "Transformer output shape mismatch"

    # Test 4: With masks
    print("\n4. Testing with Masks...")
    src_mask = create_padding_mask(src_tokens, pad_token_id=0)
    tgt_mask = create_combined_mask(tgt_tokens, pad_token_id=0, causal=True)

    print(f"   Source mask shape: {src_mask.shape}")
    print(f"   Target mask shape: {tgt_mask.shape}")

    output = transformer(src_tokens, tgt_tokens, src_mask, tgt_mask)
    print(f"   Output with masks shape: {output.shape}")

    # Test 5: Parameter count
    print("\n5. Transformer Parameters...")
    num_params = sum(p.numel() for p in transformer.parameters())
    print(f"   Total parameters: {num_params:,}")

    # Test 6: Forward pass timing
    print("\n6. Forward Pass Timing...")
    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = transformer(src_tokens, tgt_tokens, src_mask, tgt_mask)
    end = time.time()
    print(f"   Average forward pass time: {(end - start) / 10 * 1000:.2f}ms")

    print("\n✓ All Transformer tests passed!")
