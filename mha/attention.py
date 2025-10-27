"""
Multi-Head Attention (MHA) Implementation
Standard transformer attention from "Attention Is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

    Input shapes:
        Q: (batch_size, num_heads, seq_len_q, d_k)
        K: (batch_size, num_heads, seq_len_k, d_k)
        V: (batch_size, num_heads, seq_len_v, d_k)
    Output shape: (batch_size, num_heads, seq_len_q, d_k)
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, num_heads, seq_len_q, d_k)
            key: (batch_size, num_heads, seq_len_k, d_k)
            value: (batch_size, num_heads, seq_len_v, d_k)
            mask: (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, 1, 1, seq_len_k)
                  - Causal mask for decoder (prevents attending to future positions)
                  - Padding mask (prevents attending to padding tokens)

        Returns:
            output: (batch_size, num_heads, seq_len_q, d_k)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)

        # Compute attention scores: Q·K^T / √d_k
        # scores: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask (if provided)
        if mask is not None:
            # Set masked positions to large negative value (will become ~0 after softmax)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)

        # Apply attention weights to values
        # output: (batch_size, num_heads, seq_len_q, d_k)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA)

    MHA(Q, K, V) = Concat(head_1, ..., head_h)·W^O
    where head_i = Attention(Q·W^Q_i, K·W^K_i, V·W^V_i)

    Input shape:  (batch_size, seq_len, d_model)
    Output shape: (batch_size, seq_len, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)  # (d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)  # (d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)  # (d_model, d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k)

        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()

        # Reshape: (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose: (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        Combine heads back into single dimension

        Args:
            x: (batch_size, num_heads, seq_len, d_k)
        Returns:
            (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()

        # Transpose: (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)

        # Reshape: (batch_size, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask: (batch_size, seq_len_q, seq_len_k) - Optional attention mask

        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # 1. Linear projections: (batch_size, seq_len, d_model)
        Q = self.W_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.W_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.W_v(value)  # (batch_size, seq_len_v, d_model)

        # 2. Split into multiple heads: (batch_size, num_heads, seq_len, d_k)
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len_k, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len_v, d_k)

        # 3. Adjust mask dimensions if provided
        if mask is not None:
            # Add head dimension: (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)

        # 4. Apply scaled dot-product attention
        # attn_output: (batch_size, num_heads, seq_len_q, d_k)
        # attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_output, attention_weights = self.attention(Q, K, V, mask)

        # 5. Combine heads: (batch_size, seq_len_q, d_model)
        concat_output = self.combine_heads(attn_output)

        # 6. Apply output projection: (batch_size, seq_len_q, d_model)
        output = self.W_o(concat_output)

        return output, attention_weights


def create_causal_mask(seq_len, device):
    """
    Create causal mask for autoregressive modeling (decoder)

    Prevents positions from attending to future positions

    Args:
        seq_len: Sequence length
        device: torch device

    Returns:
        mask: (1, seq_len, seq_len) - Lower triangular matrix
              1 = can attend, 0 = cannot attend
    """
    # Create lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0)  # (1, seq_len, seq_len)


def create_padding_mask(seq, pad_token_id=0):
    """
    Create padding mask to prevent attention to padding tokens

    Args:
        seq: (batch_size, seq_len) - Input sequence
        pad_token_id: ID of padding token

    Returns:
        mask: (batch_size, 1, seq_len) - 1 = real token, 0 = padding
    """
    # (batch_size, seq_len) -> (batch_size, 1, seq_len)
    mask = (seq != pad_token_id).unsqueeze(1)
    return mask


def create_combined_mask(seq, pad_token_id=0, causal=True):
    """
    Create combined causal and padding mask

    Args:
        seq: (batch_size, seq_len) - Input sequence
        pad_token_id: ID of padding token
        causal: Whether to apply causal mask

    Returns:
        mask: (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = seq.size()
    device = seq.device

    # Padding mask: (batch_size, 1, seq_len)
    padding_mask = create_padding_mask(seq, pad_token_id)

    if causal:
        # Causal mask: (1, seq_len, seq_len)
        causal_mask = create_causal_mask(seq_len, device)

        # Combine: (batch_size, seq_len, seq_len)
        combined_mask = padding_mask & causal_mask
    else:
        # Just padding mask: (batch_size, seq_len, seq_len)
        combined_mask = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)

    return combined_mask


if __name__ == "__main__":
    # Unit tests for Multi-Head Attention
    print("Testing Multi-Head Attention...")

    batch_size = 2
    seq_len_q = 10
    seq_len_k = 10
    d_model = 512
    num_heads = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")

    # Test 1: ScaledDotProductAttention
    print("\n1. Testing ScaledDotProductAttention...")
    attention = ScaledDotProductAttention(dropout=0.1).to(device)

    d_k = d_model // num_heads
    Q = torch.randn(batch_size, num_heads, seq_len_q, d_k).to(device)
    K = torch.randn(batch_size, num_heads, seq_len_k, d_k).to(device)
    V = torch.randn(batch_size, num_heads, seq_len_k, d_k).to(device)

    output, attn_weights = attention(Q, K, V)
    print(f"   Q shape: {Q.shape}")
    print(f"   K shape: {K.shape}")
    print(f"   V shape: {V.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")

    # Verify attention weights sum to 1
    weights_sum = attn_weights.sum(dim=-1)
    print(f"   Attention weights sum (should be ~1.0): {weights_sum.mean().item():.6f}")
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6), "Attention weights don't sum to 1"

    # Test 2: MultiHeadAttention
    print("\n2. Testing MultiHeadAttention...")
    mha = MultiHeadAttention(d_model, num_heads, dropout=0.1).to(device)

    query = torch.randn(batch_size, seq_len_q, d_model).to(device)
    key = torch.randn(batch_size, seq_len_k, d_model).to(device)
    value = torch.randn(batch_size, seq_len_k, d_model).to(device)

    output, attn_weights = mha(query, key, value)
    print(f"   Query shape: {query.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    assert output.shape == query.shape, "MHA output shape mismatch"

    # Verify attention weights sum to 1 for each head
    weights_sum = attn_weights.sum(dim=-1)
    print(f"   Attention weights sum (should be ~1.0): {weights_sum.mean().item():.6f}")
    assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-6), "Attention weights don't sum to 1"

    # Test 3: Self-attention (Q=K=V)
    print("\n3. Testing Self-Attention...")
    x = torch.randn(batch_size, seq_len_q, d_model).to(device)
    output, attn_weights = mha(x, x, x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    assert output.shape == x.shape, "Self-attention output shape mismatch"

    # Test 4: Causal Mask
    print("\n4. Testing Causal Mask...")
    causal_mask = create_causal_mask(seq_len_q, device)
    print(f"   Causal mask shape: {causal_mask.shape}")
    print(f"   Causal mask:\n{causal_mask[0].int()}")
    output, attn_weights = mha(x, x, x, mask=causal_mask)
    print(f"   Output with causal mask shape: {output.shape}")

    # Verify causal mask prevents attending to future
    # Upper triangular should be near zero
    upper_tri = attn_weights.triu(diagonal=1)
    print(f"   Max attention to future positions: {upper_tri.max().item():.6f}")

    # Test 5: Padding Mask
    print("\n5. Testing Padding Mask...")
    seq = torch.randint(1, 100, (batch_size, seq_len_q)).to(device)
    seq[:, -3:] = 0  # Add padding at the end
    padding_mask = create_padding_mask(seq, pad_token_id=0)
    print(f"   Padding mask shape: {padding_mask.shape}")
    print(f"   Padding mask:\n{padding_mask[0]}")

    # Test 6: Combined Mask
    print("\n6. Testing Combined Mask...")
    combined_mask = create_combined_mask(seq, pad_token_id=0, causal=True)
    print(f"   Combined mask shape: {combined_mask.shape}")
    print(f"   Combined mask (first sample):\n{combined_mask[0].int()}")

    output, attn_weights = mha(x, x, x, mask=combined_mask)
    print(f"   Output with combined mask shape: {output.shape}")

    # Test 7: Count parameters
    print("\n7. MHA Parameters...")
    num_params = sum(p.numel() for p in mha.parameters())
    print(f"   Total parameters: {num_params:,}")
    print(f"   Expected: {4 * d_model * d_model:,}")

    print("\n✓ All Multi-Head Attention tests passed!")
