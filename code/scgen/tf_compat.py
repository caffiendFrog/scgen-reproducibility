"""
TensorFlow 1.x compatibility utilities for TensorFlow 2.x.

This module provides utilities to run TensorFlow 1.x style code on TensorFlow 2.x.
It automatically configures TensorFlow 2.x to disable eager execution and other
TF2 behaviors when the scgen package is imported.

For standalone scripts that import TensorFlow directly, call enable_tf1_compatibility()
explicitly before importing tensorflow or before any TensorFlow operations.
"""

import logging

log = logging.getLogger(__name__)


def enable_tf1_compatibility():
    """
    Configures TensorFlow 2.x to run TensorFlow 1.x style code.
    
    This function should be called at the very beginning of scripts/modules
    that use TensorFlow 1.x APIs (tf.placeholder, tf.Session, etc.).
    
    It:
    1. Disables TensorFlow 2.x behaviors (eager execution, etc.)
    2. Ensures tf.compat.v1 APIs are available
    3. Provides version checking and informative errors
    
    This function is idempotent - calling it multiple times is safe.
    
    Returns
    -------
    tensorflow.compat.v1
        The compatibility module, typically imported as 'tf'
    
    Raises
    ------
    ImportError
        If TensorFlow is not installed
    RuntimeError
        If TensorFlow version is incompatible or configuration fails
    
    Example
    -------
    For standalone scripts:
        from scgen.tf_compat import enable_tf1_compatibility
        enable_tf1_compatibility()
        import tensorflow as tf
    
    For code that imports scgen:
        import scgen  # Auto-configures TF compatibility
        import tensorflow as tf
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is not installed. Please install TensorFlow 2.x:\n"
            "  pip install tensorflow>=2.8,<3.0"
        ) from e
    
    # Check TensorFlow version
    tf_version = tf.__version__
    version_parts = tf_version.split('.')
    major_version = int(version_parts[0]) if version_parts[0].isdigit() else 0
    
    if major_version < 2:
        log.warning(
            f"TensorFlow version {tf_version} detected. This codebase is designed "
            "for TensorFlow 2.x. Compatibility mode may not work as expected."
        )
    
    # Enable compatibility mode
    try:
        if hasattr(tf.compat, 'v1'):
            # Disable TensorFlow 2.x behaviors (eager execution, etc.)
            tf.compat.v1.disable_v2_behavior()
            
            _patch_tf1_symbols(tf)
            
            if log.isEnabledFor(logging.INFO):
                log.info(
                    f"TensorFlow 1.x compatibility mode enabled (TF version: {tf_version}). "
                    "TF2 behaviors (eager execution) are disabled."
                )
            
            return tf.compat.v1
        else:
            raise RuntimeError(
                "tf.compat.v1 is not available. This may indicate an incompatible "
                "TensorFlow installation."
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to enable TensorFlow 1.x compatibility mode: {e}\n"
            "Please ensure TensorFlow 2.x is properly installed."
        ) from e


def get_session_config():
    """
    Returns a TF1.x Session ConfigProto with GPU-friendly defaults.

    - allow_growth: avoid pre-allocating all GPU memory
    - allow_soft_placement: fall back to CPU when an op has no GPU kernel
    """
    import tensorflow as tf
    if hasattr(tf, 'ConfigProto'):
        config = tf.ConfigProto()
    else:
        config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config


def batch_normalization(inputs, axis=-1, training=False, epsilon=1e-3, 
                        center=True, scale=True, name=None):
    """
    Compatibility wrapper for tf.layers.batch_normalization.
    
    This function provides a drop-in replacement for tf.layers.batch_normalization
    using the canonical TensorFlow 2.x compatibility API: tf.compat.v1.layers.batch_normalization.
    
    This is the official, recommended approach for maintaining backward compatibility
    with TensorFlow 1.x code in TensorFlow 2.x environments.
    
    Parameters
    ----------
    inputs : Tensor
        Input tensor to normalize
    axis : int, optional
        Integer, the axis that should be normalized (typically the features axis).
        Defaults to -1 (last axis).
    training : bool or Tensor
        Either a Python boolean or a TensorFlow boolean scalar tensor indicating
        whether the layer should behave in training mode or in inference mode.
    epsilon : float, optional
        Small float added to variance to avoid dividing by zero. Defaults to 1e-3.
    center : bool, optional
        If True, add offset of beta to normalized tensor. Defaults to True.
    scale : bool, optional
        If True, multiply by gamma. Defaults to True.
    name : str, optional
        Optional name for the operation. If None, infers from variable scope.
    
    Returns
    -------
    Tensor
        Normalized tensor with same shape as inputs
    
    Notes
    -----
    This wrapper uses tf.compat.v1.layers.batch_normalization, which is the canonical
    replacement for tf.layers.batch_normalization in TensorFlow 2.x. It maintains
    full compatibility with the original API, including:
    - Training/inference mode switching via the training parameter
    - Automatic moving average management for inference
    - Proper variable scope handling
    - Update operations added to tf.GraphKeys.UPDATE_OPS collection
    
    The function automatically infers the variable scope from context, ensuring
    proper variable reuse and naming consistent with tf.layers.batch_normalization.
    """
    import tensorflow as tf
    
    # Use the canonical compatibility layer API
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        # TensorFlow 2.x: use the official compatibility layer
        return tf.compat.v1.layers.batch_normalization(
            inputs=inputs,
            axis=axis,
            training=training,
            epsilon=epsilon,
            center=center,
            scale=scale,
            name=name
        )
    else:
        # TensorFlow 1.x: use the original API (fallback, shouldn't happen with enable_tf1_compatibility)
        return tf.layers.batch_normalization(
            inputs=inputs,
            axis=axis,
            training=training,
            epsilon=epsilon,
            center=center,
            scale=scale,
            name=name
        )


def _patch_tf1_symbols(tf):
    """
    Patch common TF1.x symbols into the top-level tf namespace.

    This keeps legacy code working after tf.compat.v1.disable_v2_behavior().
    """
    # Graph/session helpers
    if not hasattr(tf, 'reset_default_graph'):
        tf.reset_default_graph = tf.compat.v1.reset_default_graph
    if not hasattr(tf, 'placeholder'):
        tf.placeholder = tf.compat.v1.placeholder
    if not hasattr(tf, 'Session'):
        tf.Session = tf.compat.v1.Session
    if not hasattr(tf, 'ConfigProto'):
        tf.ConfigProto = tf.compat.v1.ConfigProto
    if not hasattr(tf, 'global_variables_initializer'):
        tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
    if not hasattr(tf, 'variable_scope'):
        tf.variable_scope = tf.compat.v1.variable_scope
    if not hasattr(tf, 'AUTO_REUSE'):
        tf.AUTO_REUSE = tf.compat.v1.AUTO_REUSE
    if not hasattr(tf, 'layers'):
        tf.layers = tf.compat.v1.layers

    # Optimizers/checkpoints live under compat.v1 in TF2
    if hasattr(tf.compat, 'v1') and hasattr(tf.compat.v1, 'train'):
        if not hasattr(tf, 'train') or not hasattr(tf.train, 'AdamOptimizer'):
            tf.train = tf.compat.v1.train


# Auto-configure when this module is imported
# This ensures compatibility is enabled when scgen is imported
try:
    enable_tf1_compatibility()
    import tensorflow as tf
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        _patch_tf1_symbols(tf)
except (ImportError, RuntimeError) as e:
    # Log but don't fail on import - allows the package to be imported
    # even if TensorFlow isn't installed yet (useful for documentation, etc.)
    log.warning(
        f"Could not auto-configure TensorFlow compatibility: {e}\n"
        "You may need to call enable_tf1_compatibility() explicitly."
    )
