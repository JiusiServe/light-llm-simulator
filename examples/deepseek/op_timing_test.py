"""Test indexer_wk, indexer_wq_b, scatter_nd_update, indexer_weights_proj timing on A3Pod + DS V3.2."""

from conf.config import Config
from conf.common import SEC_2_US

CONFIGS = [
    # (attn_bs, seq_len, kv_len)
    (24, 1, 2048),
    (4,  1, 4096),
    (16, 1, 4096),
    (24, 1, 4096),
    (32, 1, 4096),
    (40, 1, 4096),
    (4,  1, 16384),
    (8,  1, 16384),
    (4,  1, 32768),
    (3,  3, 65536),
]

HEADER = (
    f"{'bs':>4} {'seq_len':>7} {'kv_len':>7} | "
    f"{'op':>22} | "
    f"{'compute(us)':>12} {'memory(us)':>12} {'e2e(us)':>12}"
)
SEP = "-" * len(HEADER)

for attn_bs, seq_len, kv_len in CONFIGS:
    cfg = Config(
        serving_mode="AFD",
        model_type="deepseek-ai/DeepSeek-V3-2",
        device_type="Ascend_A3Pod",
        min_attn_bs=attn_bs,
        max_attn_bs=attn_bs,
        min_die=16,
        max_die=16,
        die_step=16,
        tpot=[200],
        kv_len=[kv_len],
        micro_batch_num=[2],
        next_n=seq_len - 1,
        multi_token_ratio=0.7,
        attn_tensor_parallel=1,
        ffn_tensor_parallel=1,
    )
    # Config stores kv_len as list; operators need the scalar value
    cfg.kv_len = kv_len

    bs = cfg.attn_bs * cfg.seq_len
    hidden = cfg.model_config.hidden_size
    indexer_dim = cfg.model_config.index_head_dim * cfg.model_config.index_n_heads
    hw = cfg.aichip_config

    from src.ops import OpMatmul, OpScatterNdUpdate

    ops = {
        "indexer_wk": OpMatmul("indexer_wk", bs, hidden, cfg.model_config.index_head_dim, hw, elem_size=2),
        "indexer_wq_b": OpMatmul("indexer_wq_b", bs, cfg.model_config.q_lora_rank, indexer_dim, hw, elem_size=2),
        "scatter_nd_update": OpScatterNdUpdate(cfg, elem_size=2),
        "indexer_weights_proj": OpMatmul("indexer_weights_proj", bs, hidden, cfg.model_config.index_n_heads, hw, elem_size=2),
        "gating_matmul": OpMatmul("gating_matmul", bs, hidden, cfg.model_config.n_routed_experts, hw, elem_size=4),
    }

    print(SEP)
    print(HEADER)
    print(SEP)
    for name, op in ops.items():
        op()
        print(
            f"{attn_bs:>4} {seq_len:>7} {kv_len:>7} | "
            f"{name:>22} | "
            f"{op.compute_time * SEC_2_US:>12.2f} {op.memory_time * SEC_2_US:>12.2f} {op.e2e_time * SEC_2_US:>12.2f}"
        )
    print()
