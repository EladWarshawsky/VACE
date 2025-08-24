# vace/models/wan/modules/base_attention.py
# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from wan.modules.model import WanAttentionBlock

class BaseWanAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        block_id=None
    ):
        # Correctly call the parent constructor to initialize self.q, self.k, self.v, etc.
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        # Get the Q, K, V, and final output tensors from the parent's forward method.
        # The variable `x_processed` holds the result of the full attention and FFN block from the parent.
        q, k, v, x_processed = super().forward(x, **kwargs)

        # Apply hint injection, which is the core purpose of this base class.
        # The hint is added to the final output of the parent block.
        if self.block_id is not None and hints is not None and self.block_id < len(hints):
            if hints[self.block_id] is not None:
                x_processed = x_processed + hints[self.block_id] * context_scale
        
        # Return all tensors for child classes to use.
        return q, k, v, x_processed
