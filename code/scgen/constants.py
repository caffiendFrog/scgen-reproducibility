"""
Constants for scgen package configuration.

This module contains shared constants used throughout the scgen package,
including default batch sizes.
"""

# Default batch size for most VAE-based models (VAE, CVAE, VAEArith)
DEFAULT_BATCH_SIZE = 512

# Batch size for ST-GAN model (typically larger due to model architecture)
STGAN_BATCH_SIZE = 4096
