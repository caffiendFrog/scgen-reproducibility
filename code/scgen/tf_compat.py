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


def batch_normalization(
    *,
    reduce_axes,
    axis=None,
    h=None,
    training=None,
    feature_dim=None,
    scope=None,
):
    """
    TF1-style batch normalization with moving averages.

    Behavior overview:
    - Creates trainable scale (gamma) and offset (beta)
    - Tracks moving_mean and moving_variance as non-trainable variables
    - During training, updates moving stats and normalizes with batch stats
    - During inference, normalizes using moving stats
    - Registers moving-stat updates in GraphKeys.UPDATE_OPS

    Calling conventions:
    - reduce_axes=True: batch_normalization(h=..., axis=..., training=...)
    - reduce_axes=False: batch_normalization(
        feature_dim=..., h=..., scope=..., training=...
      )
    """
    import tensorflow as tf

    # tensorflow vs 1.x defaults
    epsilon = 1e-3
    momentum = 0.99

    if h is None:
        raise ValueError("batch_normalization requires h.")

    if reduce_axes:
        if axis is None:
            raise ValueError("batch_normalization requires axis when reduce_axes is True.")
    else:
        if feature_dim is None:
            raise ValueError(
                "batch_normalization requires feature_dim when reduce_axes is False."
            )
        if scope is None:
            raise ValueError("batch_normalization requires scope when reduce_axes is False.")

    if feature_dim is None:
        if axis is None:
            raise ValueError(
                "batch_normalization requires axis when feature_dim is not provided."
            )
        h_shape = h.get_shape().as_list()
        axis_index = axis if axis >= 0 else (len(h_shape) + axis)
        if axis_index < 0 or axis_index >= len(h_shape):
            raise ValueError(f"batch_normalization axis {axis} is out of bounds.")
        feature_dim = h_shape[axis_index]
        if feature_dim is None:
            raise ValueError(
                "batch_normalization requires a static feature dimension to create variables."
            )

    if axis is None or not reduce_axes:
        reduce_axes_list = [0]
    else:
        h_rank = h.get_shape().ndims
        if h_rank is not None:
            axis_index = axis if axis >= 0 else (h_rank + axis)
            reduce_axes_list = [i for i in range(h_rank) if i != axis_index]
        else:
            axis_tensor = tf.convert_to_tensor(axis, dtype=tf.int32)
            rank = tf.rank(h)
            axis_tensor = tf.math.floormod(axis_tensor, rank)
            all_axes = tf.range(rank)
            mask = tf.not_equal(all_axes, axis_tensor)
            reduce_axes_list = tf.boolean_mask(all_axes, mask)

    scope_name = scope or "batch_normalization"
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        scale = tf.get_variable("scale", shape=[feature_dim], initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", shape=[feature_dim], initializer=tf.zeros_initializer())
        moving_mean = tf.get_variable(
            "moving_mean",
            shape=[feature_dim],
            initializer=tf.zeros_initializer(),
            trainable=False
        )
        moving_var = tf.get_variable(
            "moving_variance",
            shape=[feature_dim],
            initializer=tf.ones_initializer(),
            trainable=False
        )

        def _batch_norm_train():
            batch_mean, batch_var = tf.nn.moments(h, axes=reduce_axes_list)
            update_mean = tf.compat.v1.assign(
                moving_mean, moving_mean * momentum + batch_mean * (1.0 - momentum)
            )
            update_var = tf.compat.v1.assign(
                moving_var, moving_var * momentum + batch_var * (1.0 - momentum)
            )
            tf.compat.v1.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
            tf.compat.v1.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
            with tf.control_dependencies([update_mean, update_var]):
                return tf.nn.batch_normalization(
                    h, batch_mean, batch_var, offset, scale, variance_epsilon=epsilon
                )

        def _batch_norm_infer():
            return tf.nn.batch_normalization(
                h, moving_mean, moving_var, offset, scale, variance_epsilon=epsilon
            )

        if training is None:
            return _batch_norm_infer()
        if isinstance(training, bool):
            return _batch_norm_train() if training else _batch_norm_infer()
        return tf.cond(training, _batch_norm_train, _batch_norm_infer)


def dense(inputs, units, *, use_bias=True, kernel_initializer=None, kernel_regularizer=None):
    """
    TF1-style dense layer wrapper using tf.keras.layers.Dense.

    Behavior overview:
    - Mirrors tf.layers.dense argument surface used in this codebase
    - Creates a Keras Dense layer and applies it immediately
    - Supports optional bias, kernel initializer, and kernel regularizer
    - Returns the transformed tensor (no variable scope management)
    """
    import tensorflow as tf

    return tf.keras.layers.Dense(
        units=units,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
    )(inputs)


def dropout(inputs, rate, training):
    """
    TF1-style dropout wrapper with explicit training gate.

    Behavior overview:
    - Applies tf.nn.dropout only when is_training is True
    - Leaves inputs unchanged when is_training is False
    - Uses tf.cond for graph-friendly control flow
    - Expects is_training to be a boolean or boolean tensor
    """
    import tensorflow as tf

    return tf.cond(
        training,
        lambda: tf.nn.dropout(inputs, rate=rate),
        lambda: inputs,
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
    if not hasattr(tf, 'get_variable'):
        tf.get_variable = tf.compat.v1.get_variable
    if not hasattr(tf, 'get_collection'):
        tf.get_collection = tf.compat.v1.get_collection
    if not hasattr(tf, 'GraphKeys'):
        tf.GraphKeys = tf.compat.v1.GraphKeys
    if not hasattr(tf, 'AUTO_REUSE'):
        tf.AUTO_REUSE = tf.compat.v1.AUTO_REUSE
    if not hasattr(tf, 'layers'):
        tf.layers = tf.compat.v1.layers
    if not hasattr(tf, 'truncated_normal_initializer'):
        tf.truncated_normal_initializer = tf.compat.v1.truncated_normal_initializer
    if not hasattr(tf, 'assign'):
        tf.assign = tf.compat.v1.assign
    if not hasattr(tf, 'InteractiveSession'):
        tf.InteractiveSession = tf.compat.v1.InteractiveSession
    if not hasattr(tf, 'losses'):
        tf.losses = tf.compat.v1.losses

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
