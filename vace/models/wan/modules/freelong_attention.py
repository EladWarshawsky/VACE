# vace/models/wan/modules/freelong_attention.py
# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Integration of FreeLong++ logic.

import torch
import torch.nn as nn
from einops import rearrange
import math

# Import flash attention if available
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# Import the new utility functions and the block we are extending
from .freelong_utils import create_band_pass_filters, multi_band_freq_mix
from .base_attention import BaseWanAttentionBlock


class FreeLongWanAttentionBlock(BaseWanAttentionBlock):
    """
    An attention block that incorporates the FreeLong++ multi-band spectral fusion
    mechanism for generating long videos with high temporal consistency and fidelity.

    This block replaces the standard temporal attention in the main generation path.
    """

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
        block_id=None,
        # --- FreeLong++ specific arguments ---
        native_video_length=81,
        long_video_scaling_factors=[1, 2, 4], # Example for 4x length generation
        sparse_key_frame_ratio=0.5
    ):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, block_id)
        self.native_video_length = native_video_length
        self.long_video_scaling_factors = sorted(long_video_scaling_factors)
        self.sparse_key_frame_ratio = sparse_key_frame_ratio
        self.num_branches = len(self.long_video_scaling_factors)
        
        # Cache for attention masks to avoid re-computation
        self._masks_cache = {}

    def _get_attention_mask(self, sequence_length, window_size, device):
        """
        Creates and caches a sliding window attention mask.
        A frame `i` can only attend to frames in the window [i - W/2, i + W/2].
        """
        cache_key = (sequence_length, window_size)
        if cache_key in self._masks_cache:
            return self._masks_cache[cache_key].to(device)

        mask = torch.zeros((sequence_length, sequence_length), dtype=torch.bool, device=device)
        window_radius = window_size // 2

        for i in range(sequence_length):
            start = max(0, i - window_radius)
            end = min(sequence_length, i + window_radius + 1)
            mask[i, start:end] = True
        
        self._masks_cache[cache_key] = mask
        return mask

    def _sparse_key_frame_attention(self, q, k, v, sequence_length):
        """
        Performs sparse attention where queries attend only to a uniformly sampled subset of keyframes.
        This is used for the global branch to reduce computational complexity.
        """
        num_key_frames = max(1, int(sequence_length * self.sparse_key_frame_ratio))
        key_frame_indices = torch.linspace(0, sequence_length - 1, num_key_frames, dtype=torch.long, device=q.device)

        # Select only the keyframes for keys and values
        sparse_k = k[:, :, key_frame_indices, :]
        sparse_v = v[:, :, key_frame_indices, :]

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, sparse_k) * self.self_attn.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, sparse_v)
        return out

    def _efficient_windowed_attention(self, q, k, v, window_size, device):
        """Memory-efficient windowed attention using chunking"""
        b, h, s, d = q.shape
        
        if FLASH_ATTN_AVAILABLE and s > 1024:
            # Use flash attention for large sequences
            q_flat = rearrange(q, 'b h s d -> (b s) h d')
            k_flat = rearrange(k, 'b h s d -> (b s) h d') 
            v_flat = rearrange(v, 'b h s d -> (b s) h d')
            
            # Create causal mask for window
            window_radius = window_size // 2
            out = flash_attn_func(q_flat, k_flat, v_flat, 
                                causal=False, window_size=(window_radius, window_radius))
            return rearrange(out, '(b s) h d -> b h s d', b=b, s=s)
        else:
            # Fallback to chunked attention
            chunk_size = min(512, s)
            outputs = []
            
            for i in range(0, s, chunk_size):
                end_i = min(i + chunk_size, s)
                q_chunk = q[:, :, i:end_i]
                
                # Determine attention range for this chunk
                window_radius = window_size // 2
                k_start = max(0, i - window_radius)
                k_end = min(s, end_i + window_radius)
                
                k_chunk = k[:, :, k_start:k_end]
                v_chunk = v[:, :, k_start:k_end]
                
                sim = torch.einsum('b h i d, b h j d -> b h i j', q_chunk, k_chunk) * self.self_attn.scale
                attn = sim.softmax(dim=-1)
                out_chunk = torch.einsum('b h i j, b h j d -> b h i d', attn, v_chunk)
                outputs.append(out_chunk)
                
            return torch.cat(outputs, dim=2)

    def forward(self, x, hints=None, context_scale=1.0, **kwargs):
        """
        Overrides the standard attention block's forward pass to implement FreeLong++.
        This method now correctly uses the inherited tensors from its parent.
        """
        # 1. Get tensors from parent.
        # q, k, v are the raw query, key, value tensors.
        # x_processed is the final output from the parent block, after its attention and FFN.
        q, k, v, x_processed = super().forward(x, hints=hints, context_scale=context_scale, **kwargs)

        # Reshape for multi-head attention.
        q, k, v = map(lambda t: rearrange(t, 'b s (n d) -> b n s d', n=self.num_heads), (q, k, v))

        num_patches_spatial = x.shape[0]
        video_length = x.shape[1]

        # 2. Multi-Scale Attention Decoupling (FreeLong++ logic)
        branch_features = []
        for i, scale_factor in enumerate(self.long_video_scaling_factors):
            is_global_branch = (i == self.num_branches - 1)
            window_size = min(video_length, int(self.native_video_length * scale_factor))

            if is_global_branch and self.sparse_key_frame_ratio < 1.0 and video_length > self.native_video_length:
                out = self._sparse_key_frame_attention(q, k, v, video_length)
            else:
                # Use efficient windowed attention instead of full attention matrix
                out = self._efficient_windowed_attention(q, k, v, window_size, x.device)

            branch_out = rearrange(out, 'b n s d -> b s (n d)')
            branch_features.append(branch_out)
            
            # Clear intermediate tensors to save memory
            del out

        # 3. Multi-band Spectral Fusion
        if len(branch_features) > 1:
            h_patches = w_patches = int(math.sqrt(num_patches_spatial))
            if h_patches * w_patches != num_patches_spatial:
                raise ValueError("Spatial patch dimension is not a perfect square.")

            reshaped_features = [rearrange(feat, '(h w) t c -> 1 c t h w', h=h_patches) for feat in branch_features]
            filters = create_band_pass_filters(self.num_branches, (video_length, h_patches, w_patches), x.device)
            fused_features = multi_band_freq_mix(reshaped_features, filters)
            x_fused = rearrange(fused_features.squeeze(0), 'c t h w -> (h w) t c')
            
            # Clear intermediate tensors
            del reshaped_features, filters, fused_features
        else:
            x_fused = branch_features[0]

        # Clear branch features
        del branch_features

        # 4. Final Output Construction
        # The parent's `x_processed` is the result of the full block (self-attn + cross-attn + ffn).
        # Our `x_fused` is the output of our custom self-attention.
        # We need to reconstruct the final output by replacing the original self-attention result
        # with our fused result, while keeping the cross-attention and FFN parts.

        # Get the original self-attention output from the grandparent's forward pass
        e = kwargs.get('e')
        norm_x = self.norm1(x).float() * (1 + e[1]) + e[0]
        original_self_attn_out = self.self_attn(norm_x, **{k: v for k, v in kwargs.items() if k in ['seq_lens', 'grid_sizes', 'freqs']})
        
        # The parent's output is: x_processed = x + original_self_attn_out + cross_attn + ffn
        # We want: final_output = x + new_self_attn_out + cross_attn + ffn
        # So, final_output = x_processed - original_self_attn_out + new_self_attn_out
        
        # Project our fused features
        new_self_attn_out = self.self_attn.o(x_fused)

        # Reconstruct the final output
        final_output = x_processed - original_self_attn_out + new_self_attn_out

        return final_output
