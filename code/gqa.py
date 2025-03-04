from typing import Optional

import torch
from torch.testing import assert_close


def gqa(
    x: torch.Tensor,  # shape: (batch_size, seq_len, dim)
    num_query_heads: int,  # number of query heads
    num_kv_heads: int,  # number of key-value groups
    head_dim: int,  # dimension of each head
    w_q: torch.Tensor,  # shape: (num_query_heads * head_dim, dim)
    w_k: torch.Tensor,  # shape: (num_kv_heads * head_dim, dim)
    w_v: torch.Tensor,  # shape: (num_kv_heads * head_dim, dim)
    w_o: torch.Tensor,  # shape: (dim, num_query_heads * head_dim)
    mask: Optional[torch.Tensor] = None,  # shape: (batch_size, seq_len, seq_len)
) -> torch.Tensor:  # shape: (batch_size, seq_len, dim)
    """
    Compute Grouped Query Attention.

    The mask, if provided, will be expanded to shape (batch_size, num_query_heads, seq_len, seq_len) to
        properly broadcast across attention heads.

    Returns:
        Output tensor of shape (batch_size, seq_len, dim)
    """

    # Your implementation here
    # 1. Project input to queries, keys, and values using provided weight matrices
    # 2. Reshape projections to separate heads
    # 3. Repeat keys and values for each query group
    # 4. Apply scaled dot-product attention
    # 5. Reshape and project output

    pass


def test_gqa():
    # Test Case 1: Basic functionality
    batch_size = 2
    seq_len = 4
    dim = 6
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 3

    # Create input and weights
    x = torch.randn(batch_size, seq_len, dim)
    w_q = torch.randn(num_query_heads * head_dim, dim)
    w_k = torch.randn(num_kv_heads * head_dim, dim)
    w_v = torch.randn(num_kv_heads * head_dim, dim)
    w_o = torch.randn(dim, num_query_heads * head_dim)

    output = gqa(x, num_query_heads, num_kv_heads, head_dim, w_q, w_k, w_v, w_o)
    assert output.shape == (batch_size, seq_len, dim)

    # Test Case 2: With masking
    mask = torch.zeros(batch_size, seq_len, seq_len)
    mask[:, -1, :] = 1  # Mask last position
    output_masked = gqa(
        x, num_query_heads, num_kv_heads, head_dim, w_q, w_k, w_v, w_o, mask
    )
    assert output_masked.shape == (batch_size, seq_len, dim)

    # Test Case 3: Compare with reference implementation
    x, w_q, w_k, w_v, w_o, num_query_heads, num_kv_heads, head_dim = torch.load(
        "gqa_input.pt"
    )
    output = gqa(x, num_query_heads, num_kv_heads, head_dim, w_q, w_k, w_v, w_o)

    # Load reference output
    reference = torch.load("gqa_ref_output.pt")
    assert_close(output, reference, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_gqa()
    print("All tests passed!")
