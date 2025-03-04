from typing import Optional, Tuple

import torch
from torch.testing import assert_close


def scaled_dot_product_attention(
    query: torch.Tensor,  # shape: (batch_size, num_heads, seq_len_q, depth)
    key: torch.Tensor,  # shape: (batch_size, num_heads, seq_len_k, depth)
    value: torch.Tensor,  # shape: (batch_size, num_heads, seq_len_v, depth)
    mask: Optional[
        torch.Tensor
    ] = None,  # shape: (batch_size, num_heads, seq_len_q, seq_len_k)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention with optional masking.

    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        mask: Optional mask tensor. Should contain 0s for valid positions and 1s for masked positions.
            Will be converted to -inf for masked positions before softmax.

    Returns:
        tuple: (
            output tensor with shape (batch_size, num_heads, seq_len_q, depth),
            attention weights with shape (batch_size, num_heads, seq_len_q, seq_len_k)
        )

    Example:
        >>> q = torch.randn(2, 3, 4, 8)  # batch=2, heads=3, seq_len=4, depth=8
        >>> k = torch.randn(2, 3, 4, 8)
        >>> v = torch.randn(2, 3, 4, 8)
        >>> mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()  # causal mask
        >>> mask = mask.expand(2, 3, 4, 4)  # expand to match batch and heads
        >>> output, attention = scaled_dot_product_attention(q, k, v, mask)
        >>> print(output.shape)  # (2, 3, 4, 8)
        >>> print(attention.shape)  # (2, 3, 4, 4)
    """

    # Your implementation here
    # 1. Scale the attention scores by $\sqrt{d_k}$
    # 2. Apply the mask (if provided) before softmax
    # 3. Implement the attention computation efficiently using matrix operations
    # 4. Return both the output and attention weights

    pass


def test_basic_attention():
    batch_size, num_heads, seq_len, depth = 2, 3, 4, 8
    query = torch.randn(batch_size, num_heads, seq_len, depth)
    key = torch.randn(batch_size, num_heads, seq_len, depth)
    value = torch.randn(batch_size, num_heads, seq_len, depth)

    # Custom implementation
    custom_output, custom_weights = scaled_dot_product_attention(query, key, value)

    # PyTorch implementation (flip mask convention not needed as no mask used)
    torch_output = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    # Assert outputs are close
    assert_close(custom_output, torch_output)
    assert custom_output.shape == (batch_size, num_heads, seq_len, depth)
    assert custom_weights.shape == (batch_size, num_heads, seq_len, seq_len)


def test_padding_mask():
    batch_size, num_heads, seq_len, depth = 2, 3, 4, 8
    query = torch.randn(batch_size, num_heads, seq_len, depth)
    key = torch.randn(batch_size, num_heads, seq_len, depth)
    value = torch.randn(batch_size, num_heads, seq_len, depth)

    # Create padding mask (1s for padded positions)
    padding_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
    padding_mask[..., -1:] = 1  # Mask last position

    # Custom implementation
    custom_output, custom_weights = scaled_dot_product_attention(
        query, key, value, padding_mask
    )

    # PyTorch implementation (flip mask: True means attend, False means mask)
    torch_mask = ~padding_mask.bool()  # Flip mask convention
    torch_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=torch_mask
    )

    assert_close(custom_output, torch_output)


def test_causal_mask():
    batch_size, num_heads, seq_len, depth = 2, 3, 4, 8
    query = torch.randn(batch_size, num_heads, seq_len, depth)
    key = torch.randn(batch_size, num_heads, seq_len, depth)
    value = torch.randn(batch_size, num_heads, seq_len, depth)

    # Create causal mask (1s for future positions)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    causal_mask = causal_mask.expand(batch_size, num_heads, seq_len, seq_len)

    # Custom implementation
    custom_output, custom_weights = scaled_dot_product_attention(
        query, key, value, causal_mask
    )

    # PyTorch implementation (flip mask: True means attend, False means mask)
    torch_mask = ~causal_mask.bool()  # Flip mask convention
    torch_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=torch_mask
    )

    assert_close(custom_output, torch_output)


def test_combined_mask():
    batch_size, num_heads, seq_len, depth = 2, 3, 4, 8
    query = torch.randn(batch_size, num_heads, seq_len, depth)
    key = torch.randn(batch_size, num_heads, seq_len, depth)
    value = torch.randn(batch_size, num_heads, seq_len, depth)

    # Create causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    causal_mask = causal_mask.expand(batch_size, num_heads, seq_len, seq_len)

    # Create padding mask
    padding_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len)
    padding_mask[..., -1:] = 1  # Mask last position

    # Combine masks (1 in either mask means position should be masked)
    combined_mask = torch.logical_or(causal_mask, padding_mask)

    # Custom implementation
    custom_output, custom_weights = scaled_dot_product_attention(
        query, key, value, combined_mask
    )

    # PyTorch implementation (flip mask: True means attend, False means mask)
    torch_mask = ~combined_mask.bool()  # Flip mask convention
    torch_output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=torch_mask
    )

    assert_close(custom_output, torch_output)


def test_edge_cases():
    # Test with sequence length of 1
    batch_size, num_heads, depth = 2, 3, 8
    seq_len = 1
    query = torch.randn(batch_size, num_heads, seq_len, depth)
    key = torch.randn(batch_size, num_heads, seq_len, depth)
    value = torch.randn(batch_size, num_heads, seq_len, depth)

    custom_output, custom_weights = scaled_dot_product_attention(query, key, value)
    torch_output = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    assert_close(custom_output, torch_output)

    # Test with different key/value sequence length
    seq_len_q, seq_len_kv = 4, 6
    query = torch.randn(batch_size, num_heads, seq_len_q, depth)
    key = torch.randn(batch_size, num_heads, seq_len_kv, depth)
    value = torch.randn(batch_size, num_heads, seq_len_kv, depth)

    custom_output, custom_weights = scaled_dot_product_attention(query, key, value)
    torch_output = torch.nn.functional.scaled_dot_product_attention(query, key, value)

    assert_close(custom_output, torch_output)
    assert custom_output.shape == (batch_size, num_heads, seq_len_q, depth)
    assert custom_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_kv)


if __name__ == "__main__":
    test_basic_attention()
    test_padding_mask()
    test_causal_mask()
    test_combined_mask()
    test_edge_cases()

    print("All tests passed!")
