from modelopt.torch.puzzletron.utils.misc import calculate_kv_dim


def test_calculate_kv_dim_with_explicit_head_dim():
    assert calculate_kv_dim(
        num_key_value_heads=8,
        n_head=32,
        n_embd=5120,
        head_dim=128,
    ) == 2048


def test_calculate_kv_dim_falls_back_to_hidden_size():
    assert calculate_kv_dim(
        num_key_value_heads=8,
        n_head=32,
        n_embd=4096,
    ) == 2048