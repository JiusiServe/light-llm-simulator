import math
from src.ops.base import BaseOp
from conf.common import US_2_SEC, SPARSE_COUNT


class OpLightningIndexer(BaseOp):
    '''
    Description:
        The Lightning Indexer operation for sparse attention.
        Computes Top-k positions for each query token based on index Q/K similarity.

    Attributes:
        bs: Batch size.
        seq_len: Query sequence length (S1).
        kv_len: Key/Value length (S2).
        num_head: Number of query heads (N1).
        kv_heads: Number of key heads (N2).
        head_size: Head dimension (D).
    '''

    def __init__(self, name, config, elem_size=2):
        self.aichip_config = config.aichip_config
        self.static_cost = 30 * US_2_SEC
        super().__init__(name, self.aichip_config, elem_size, static_cost=self.static_cost)
        self.bs = config.attn_bs
        self.seq_len = config.seq_len
        self.kv_len = config.kv_len
        self.num_head = config.model_config.num_attention_heads
        self.kv_heads = 1
        self.head_size = config.model_config.qk_nope_head_dim

        # GQA group size: g = N1 / N2
        self.g_size = self.num_head // self.kv_heads

    def compute_cost(self):
        '''
        Computational formula:
            Indices = Top-k { [1]_{1xg} @ [ (W @ [1]_{1xSk}) ⊙ ReLU(Q_index @ K_index^T) ] }

        Computation stages:
            1. QK Matmul (cube):  Q_index @ K_index^T  → [B, S1, g, S2]
            2. ReLU + Scale (vector): ReLU(QK) ⊙ W_broadcast  → [B, S1, g, S2]
            3. Group Reduce (vector): sum over g dimension  → [B, S1, S2]
            4. TopK (vector): select top-sparseCount indices  → [B, S1, N2, sparseCount]
        '''
        g = self.g_size

        # Stage 1: QK Matmul (cube core)
        # Q_index: [B, S1, g, D], K_index: [B, S2, N2, D] -> [B, S1, g, S2]
        # FLOPS = 2 * B * S1 * g * D * S2
        qk_flops = 2 * self.bs * self.seq_len * g * self.head_size * self.kv_len
        qk_time = qk_flops / self.cube_flops

        # Stage 2: ReLU + Element-wise Scale (vector core)
        # Weights W: [B, S1, N1] -> broadcast to [B, S1, g, S2]
        # ReLU + multiply: 2 * B * S1 * g * S2
        scale_flops = 2 * self.bs * self.seq_len * g * self.kv_len
        scale_time = scale_flops / self.vector_flops

        # Stage 3: Group Reduce - sum over g dimension (vector core)
        # [1]_{1xg} @ [B, S1, g, S2] -> [B, S1, S2]
        # g-1 additions per output element: B * S1 * S2 * (g - 1)
        reduce_flops = self.bs * self.seq_len * self.kv_len * (g - 1)
        reduce_time = reduce_flops / self.vector_flops

        # Stage 4: TopK via merge sort (vector core)
        # Per (B, S1, N2) element, select top-k from S2 values
        # Merge sort based TopK: O(S2 * log2(S2)) per element
        topk_flops = self.bs * self.seq_len * self.kv_heads * self.kv_len * math.ceil(math.log2(max(self.kv_len, 2)))
        topk_time = topk_flops / self.vector_flops

        self.total_computation = qk_flops + scale_flops + reduce_flops + topk_flops
        self.compute_time = qk_time + scale_time + reduce_time + topk_time
        return self.compute_time

    def memory_cost(self):
        # Input: query [B, S1, N1, D] bf16
        query_bytes = self.bs * self.seq_len * self.num_head * self.head_size * self.elem_size
        # Input: key [B, S2, N2, D] bf16
        key_bytes = self.bs * self.kv_len * self.kv_heads * self.head_size * self.elem_size
        # Input: weights [B, S1, N1] bf16
        weights_bytes = self.bs * self.seq_len * self.num_head * self.elem_size
        # Output: sparse_indices [B, S1, N2, sparseCount] INT32
        indices_bytes = self.bs * self.seq_len * self.kv_heads * SPARSE_COUNT * 4

        self.total_data_movement = query_bytes + key_bytes + weights_bytes + indices_bytes
        self.memory_time = self.total_data_movement / self.local_memory_bandwidth
        return self.memory_time
