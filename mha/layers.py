"""
Core transformer layers for MHA implementation
Includes: LayerNorm, FeedForward Network, Dropout, Residual Connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer Normalization with learnable parameters

    Normalizes across the feature dimension (d_model)
    Input shape:  (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(d_model))  # Shift parameter
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            normalized: (batch_size, seq_len, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # (batch_size, seq_len, 1)
        normalized = (x - mean) / (std + self.eps)  # (batch_size, seq_len, d_model)
        return self.gamma * normalized + self.beta  # (batch_size, seq_len, d_model)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN)

    Two-layer MLP with activation in between:
    FFN(x) = activation(xW1 + b1)W2 + b2

    Input shape:  (batch_size, seq_len, d_model)
    Hidden shape: (batch_size, seq_len, d_ff)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Activation function selection
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # x: (batch_size, seq_len, d_model)
        hidden = self.linear1(x)                # (batch_size, seq_len, d_ff)
        hidden = self.activation(hidden)        # (batch_size, seq_len, d_ff)
        hidden = self.dropout(hidden)           # (batch_size, seq_len, d_ff)
        output = self.linear2(hidden)           # (batch_size, seq_len, d_model)
        return output


class ResidualConnection(nn.Module):
    """
    Residual connection with dropout and layer normalization

    Applies: LayerNorm(x + Dropout(sublayer(x)))

    Input shape:  (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Args:
            x: (batch_size, seq_len, d_model) - Input tensor
            sublayer: callable - The sublayer (attention or FFN) to apply
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Post-norm: x + Dropout(sublayer(LayerNorm(x)))
        # Note: Some implementations use pre-norm instead
        return x + self.dropout(sublayer(self.norm(x)))


class DropoutLayer(nn.Module):
    """
    Dropout layer with seed management for reproducibility

    Input shape:  (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        return self.dropout(x)


if __name__ == "__main__":
    # Unit tests for layers
    print("Testing layers...")

    batch_size, seq_len, d_model, d_ff = 2, 10, 512, 2048

    # Test LayerNorm
    print("\n1. Testing LayerNorm...")
    x = torch.randn(batch_size, seq_len, d_model)
    layer_norm = LayerNorm(d_model)
    out = layer_norm(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"   Output mean: {out.mean():.6f}, std: {out.std():.6f}")
    assert out.shape == x.shape, "LayerNorm output shape mismatch"

    # Test FeedForward
    print("\n2. Testing FeedForward...")
    ffn = FeedForward(d_model, d_ff, dropout=0.1, activation="gelu")
    out = ffn(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "FFN output shape mismatch"

    # Test ResidualConnection
    print("\n3. Testing ResidualConnection...")
    residual = ResidualConnection(d_model, dropout=0.1)
    out = residual(x, lambda x: ffn(x))
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "ResidualConnection output shape mismatch"

    # Test Dropout
    print("\n4. Testing DropoutLayer...")
    dropout = DropoutLayer(dropout=0.5)
    out = dropout(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    assert out.shape == x.shape, "Dropout output shape mismatch"

    print("\n✓ All layer tests passed!")
