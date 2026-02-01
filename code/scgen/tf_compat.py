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


def _get_legacy_layer(tf, layer_name):
    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
        v1_layers = getattr(tf.compat.v1, 'layers', None)
        if v1_layers is not None and hasattr(v1_layers, layer_name):
            return getattr(v1_layers, layer_name)
    if hasattr(tf, 'layers') and hasattr(tf.layers, layer_name):
        return getattr(tf.layers, layer_name)
    return None


def _call_keras_layer(layer_cls, inputs, training=None, **kwargs):
    layer = layer_cls(**kwargs)
    if training is None:
        return layer(inputs)
    return layer(inputs, training=training)


def batch_normalization(scope, feature_dim, h, training):
    """
    Manual batch normalization workaround for TensorFlow 2.x compatibility.

    How it works:
    1. Creates trainable scale (gamma) and offset (beta) variables
    2. Computes batch mean and variance using tf.nn.moments
    3. Applies normalization: (x - mean) / sqrt(variance + epsilon) * scale + offset

    This is equivalent to tf.layers.batch_normalization but uses low-level TF1.x APIs
    that are still available in TF2.x via compat.v1.
    """
    import tensorflow as tf

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        scale = tf.get_variable("scale", shape=[feature_dim], initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", shape=[feature_dim], initializer=tf.zeros_initializer())
        batch_mean, batch_var = tf.nn.moments(h, axes=[0])
        return tf.nn.batch_normalization(h, batch_mean, batch_var, offset, scale, variance_epsilon=1e-5)


def dense(inputs, units, activation=None, use_bias=True, kernel_initializer=None,
          bias_initializer=None, kernel_regularizer=None, bias_regularizer=None,
          activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
          name=None, **kwargs):
    """
    Compatibility wrapper for tf.layers.dense with Keras 3 fallback.
    """
    import tensorflow as tf
    legacy = _get_legacy_layer(tf, 'dense')
    if legacy is not None:
        return legacy(
            inputs=inputs,
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            **kwargs
        )
    return _call_keras_layer(
        tf.keras.layers.Dense,
        inputs,
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        name=name
    )


def dropout(inputs, rate, training=False, noise_shape=None, seed=None, name=None):
    """
    Compatibility wrapper for tf.layers.dropout with Keras 3 fallback.
    """
    import tensorflow as tf
    legacy = _get_legacy_layer(tf, 'dropout')
    if legacy is not None:
        return legacy(
            inputs=inputs,
            rate=rate,
            training=training,
            noise_shape=noise_shape,
            seed=seed,
            name=name
        )
    return _call_keras_layer(
        tf.keras.layers.Dropout,
        inputs,
        training=training,
        rate=rate,
        noise_shape=noise_shape,
        seed=seed,
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
    if not hasattr(tf, 'truncated_normal_initializer'):
        tf.truncated_normal_initializer = tf.compat.v1.truncated_normal_initializer

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
