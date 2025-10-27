"""
Positional Encoding implementations for MHA Transformer
Includes: Sinusoidal (fixed) and Learned positional encodings
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need"

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Input shape:  (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # Create positional encoding matrix
        # pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # position: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: (d_model/2,)
        # Compute the positional encodings once in log space for numerical stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model/2)

        # Apply cos to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model/2)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter, but part of module state)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) - Input embeddings
        Returns:
            output: (batch_size, seq_len, d_model) - Input + positional encoding
        """
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)

        # Add positional encoding to input
        # pe[:, :seq_len, :]: (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding

    Uses an embedding layer to learn position representations
    Often used in BERT and other transformer variants

    Input shape:  (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Learnable position embeddings
        # (max_len, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model) - Input embeddings
        Returns:
            output: (batch_size, seq_len, d_model) - Input + positional encoding
        """
        batch_size, seq_len, d_model = x.size()

        # Create position indices: (seq_len,)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)

        # Get position embeddings: (seq_len, d_model)
        position_encodings = self.position_embeddings(positions)

        # Add to input (broadcasting across batch): (batch_size, seq_len, d_model)
        x = x + position_encodings.unsqueeze(0)

        return self.dropout(x)


class PositionalEncodingFactory:
    """
    Factory to create positional encoding based on type
    """
    @staticmethod
    def create(pe_type: str, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            pe_type: "sinusoidal" or "learned"
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability

        Returns:
            Positional encoding module
        """
        if pe_type == "sinusoidal":
            return SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif pe_type == "learned":
            return LearnedPositionalEncoding(d_model, max_len, dropout)
        else:
            raise ValueError(f"Unknown positional encoding type: {pe_type}")


if __name__ == "__main__":
    # Unit tests for positional encodings
    print("Testing positional encodings...")

    batch_size, seq_len, d_model = 2, 10, 512

    # Test Sinusoidal Positional Encoding
    print("\n1. Testing SinusoidalPositionalEncoding...")
    x = torch.randn(batch_size, seq_len, d_model)
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model, max_len=1000)
    out = sinusoidal_pe(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "Sinusoidal PE output shape mismatch"

    # Verify positional encoding is deterministic (same for all batches)
    out1 = sinusoidal_pe(x)
    out2 = sinusoidal_pe(x)
    print(f"   Deterministic check: {torch.allclose(out1, out2, atol=1e-6)}")

    # Test Learned Positional Encoding
    print("\n2. Testing LearnedPositionalEncoding...")
    learned_pe = LearnedPositionalEncoding(d_model, max_len=1000)
    out = learned_pe(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "Learned PE output shape mismatch"
    print(f"   Num parameters: {sum(p.numel() for p in learned_pe.parameters())}")

    # Test Factory
    print("\n3. Testing PositionalEncodingFactory...")
    pe_sin = PositionalEncodingFactory.create("sinusoidal", d_model)
    pe_learned = PositionalEncodingFactory.create("learned", d_model)
    print(f"   Created sinusoidal: {type(pe_sin).__name__}")
    print(f"   Created learned: {type(pe_learned).__name__}")

    # Test with different sequence lengths
    print("\n4. Testing variable sequence lengths...")
    for test_seq_len in [5, 50, 500]:
        x_test = torch.randn(batch_size, test_seq_len, d_model)
        out_sin = sinusoidal_pe(x_test)
        out_learned = learned_pe(x_test)
        assert out_sin.shape == x_test.shape, f"Failed for seq_len={test_seq_len}"
        assert out_learned.shape == x_test.shape, f"Failed for seq_len={test_seq_len}"
    print(f"   ✓ All sequence lengths work correctly")

    print("\n✓ All positional encoding tests passed!")
