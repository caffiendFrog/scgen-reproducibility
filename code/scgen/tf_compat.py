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


# Auto-configure when this module is imported
# This ensures compatibility is enabled when scgen is imported
try:
    enable_tf1_compatibility()
except (ImportError, RuntimeError) as e:
    # Log but don't fail on import - allows the package to be imported
    # even if TensorFlow isn't installed yet (useful for documentation, etc.)
    log.warning(
        f"Could not auto-configure TensorFlow compatibility: {e}\n"
        "You may need to call enable_tf1_compatibility() explicitly."
    )
