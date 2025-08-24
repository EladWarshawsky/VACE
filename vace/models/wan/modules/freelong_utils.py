# vace/models/wan/modules/freelong_utils.py
# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# Utilities adapted from FreeLong++ for spectral fusion.

import torch
import torch.fft as fft

def create_band_pass_filters(num_bands: int, shape: tuple, device: torch.device):
    """
    Creates a series of ideal band-pass filters in the frequency domain.
    This is a simplified implementation where frequency bands are divided linearly.

    Args:
        num_bands (int): The number of frequency bands to create (should match the number of attention branches).
        shape (tuple): The shape of the spatial-temporal volume (T, H, W).
        device (torch.device): The torch device to create tensors on.

    Returns:
        list[torch.Tensor]: A list of band-pass filter masks.
    """
    T, H, W = shape
    
    # Create frequency grid coordinates
    freq_t = torch.fft.fftfreq(T, device=device)
    freq_h = torch.fft.fftfreq(H, device=device)
    freq_w = torch.fft.fftfreq(W, device=device)
    
    # Generate a meshgrid of frequency coordinates
    ft, fh, fw = torch.meshgrid(freq_t, freq_h, freq_w, indexing='ij')
    
    # Calculate the magnitude of the frequency vector (distance from origin in frequency space)
    # This represents the "overall" frequency of each component.
    freq_mag = torch.sqrt(ft**2 + fh**2 + fw**2)
    
    # Define frequency band cutoffs by linearly spacing them.
    # A more advanced implementation could link these cutoffs to the Nyquist frequency
    # of each attention window size, as suggested in the paper.
    max_freq = freq_mag.max()
    cutoffs = torch.linspace(0, max_freq, num_bands + 1, device=device)
    
    filters = []
    for i in range(num_bands):
        lower_bound = cutoffs[i]
        upper_bound = cutoffs[i+1]
        
        # Create a boolean mask for the current band
        band_mask = (freq_mag >= lower_bound) & (freq_mag < upper_bound)
        
        # Ensure the last band includes the highest frequency component
        if i == num_bands - 1:
            band_mask = (freq_mag >= lower_bound)

        filters.append(band_mask.float())
        
    return filters

def multi_band_freq_mix(features: list[torch.Tensor], filters: list[torch.Tensor]):
    """
    Fuses multiple feature sets in the frequency domain using pre-computed band-pass filters.

    Args:
        features (list[torch.Tensor]): A list of feature tensors from each attention branch.
                                       The shape of each tensor should be (N, C, T, H, W).
        filters (list[torch.Tensor]): A list of corresponding band-pass filters.

    Returns:
        torch.Tensor: The fused feature tensor, transformed back into the spatial domain.
    """
    if not features:
        return None
        
    # Initialize a complex tensor to store the fused frequency representation
    fused_freq = torch.zeros_like(torch.fft.rfftn(features[0].float(), dim=(-3, -2, -1)), dtype=torch.complex64)

    for i, feat in enumerate(features):
        # Transform the feature from spatial to frequency domain using real-to-complex FFT
        feat_freq = torch.fft.rfftn(feat.float(), dim=(-3, -2, -1))
        
        # Get the corresponding filter for this branch
        current_filter = filters[i]
        # Adjust filter shape for broadcasting (rfftn's last dim is smaller)
        current_filter = current_filter[..., :feat_freq.shape[-1]]
        
        # Apply the band-pass filter to isolate the desired frequencies
        filtered_component = feat_freq * current_filter
        
        # Add the filtered component to the combined frequency representation
        fused_freq += filtered_component

    # Transform the fused frequency representation back to the spatial domain
    fused_spatial = torch.fft.irfftn(fused_freq, s=features[0].shape[-3:]).to(features[0].dtype)
    
    return fused_spatial

import math
import random

def specmix_initialization(shape: tuple, device: torch.device):
    """
    Implements SpecMix noise initialization for long video generation.
    Blends a consistent base noise with a random residual noise in the frequency domain.
    
    Args:
        shape (tuple): The shape of the noise tensor (B, C, T, H, W).
        device (torch.device): The device to create tensors on.

    Returns:
        torch.Tensor: The initialized noise tensor.
    """
    B, C, T, H, W = shape
    
    # 1. Create the consistent base noise (x_base) using sliding-window shuffling
    base_noise = torch.randn(shape, device=device)
    window_size = 16  # A common window size for consistency
    stride = 4      # A common stride
    if T > window_size:
        for i in range(window_size, T, stride):
            # Define a window and shuffle indices within it
            window_indices = list(range(max(0, i - window_size), i))
            shuffled_indices = list(range(max(0, i - window_size), i))
            random.shuffle(shuffled_indices)
            
            # Apply shuffle to a segment
            segment_to_shuffle = base_noise[:, :, shuffled_indices, :, :]
            base_noise[:, :, window_indices, :, :] = segment_to_shuffle

    # 2. Create the per-frame residual noise (x_res)
    residual_noise = torch.randn(shape, device=device)
    
    # 3. Transform both to the frequency domain
    base_freq = torch.fft.rfftn(base_noise.float(), dim=(-3, -2, -1))
    residual_freq = torch.fft.rfftn(residual_noise.float(), dim=(-3, -2, -1))
    
    # 4. Blend based on temporal position
    # Create a mixing angle that goes from pi/2 at the edges to 0 in the center
    time_indices = torch.arange(T, device=device)
    center_dist = torch.abs(time_indices - (T - 1) / 2)
    normalized_dist = center_dist / ((T - 1) / 2)
    mix_angle = (normalized_dist * (math.pi / 2)).view(1, 1, T, 1, 1) # Shape for broadcasting

    cos_mix = torch.cos(mix_angle)
    sin_mix = torch.sin(mix_angle)
    
    # Blend: center frames use more base_freq, edge frames use more residual_freq
    mixed_freq = base_freq * cos_mix + residual_freq * sin_mix
    
    # 5. Transform back to spatial domain
    mixed_noise = torch.fft.irfftn(mixed_freq, s=(T, H, W))
    
    return mixed_noise.to(base_noise.dtype)
