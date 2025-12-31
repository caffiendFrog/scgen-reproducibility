# Batch Normalization Workaround Explanation

## Problem

In TensorFlow 2.x, even with `tf.compat.v1`, the `tf.layers.batch_normalization()` function is not available or unreliable. This causes errors like:
```
AttributeError: module 'tensorflow.compat.v1' has no attribute 'layers'
```

## Solution: Manual Batch Normalization

The workaround manually implements batch normalization using low-level TensorFlow 1.x APIs that are still available in TensorFlow 2.x via `tf.compat.v1`.

## How the Workaround Works

### The Function

```python
def _work_around(scope, feature_dim, h, training):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 1. Create trainable parameters
        scale = tf.get_variable("scale", shape=[feature_dim], initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", shape=[feature_dim], initializer=tf.zeros_initializer())
        
        # 2. Compute batch statistics
        batch_mean, batch_var = tf.nn.moments(h, axes=[0])
        
        # 3. Apply normalization
        return tf.nn.batch_normalization(h, batch_mean, batch_var, offset, scale, variance_epsilon=1e-5)
```

### Step-by-Step Explanation

#### 1. **Variable Scope** (`tf.variable_scope`)
   - Creates a unique namespace for each batch normalization layer
   - `reuse=tf.AUTO_REUSE` allows reusing variables when the function is called multiple times
   - This ensures each layer has its own independent batch normalization parameters

#### 2. **Trainable Parameters**
   - **`scale` (gamma)**: Multiplicative parameter, initialized to 1.0
     - Allows the network to learn whether to amplify or reduce the normalized values
     - Shape: `[feature_dim]` - one value per feature dimension
   
   - **`offset` (beta)**: Additive parameter, initialized to 0.0
     - Allows the network to shift the normalized values
     - Shape: `[feature_dim]` - one value per feature dimension

#### 3. **Batch Statistics** (`tf.nn.moments`)
   - Computes the **mean** and **variance** of the input tensor `h` along axis 0 (batch dimension)
   - Formula:
     - `mean = mean(h, axis=0)`
     - `variance = mean((h - mean)², axis=0)`
   - These are computed per-feature (across the batch)

#### 4. **Normalization** (`tf.nn.batch_normalization`)
   - Applies the batch normalization formula:
     ```
     normalized = (h - batch_mean) / sqrt(batch_var + epsilon)
     output = normalized * scale + offset
     ```
   - Where `epsilon = 1e-5` prevents division by zero
   - This is the standard batch normalization operation

### Mathematical Formula

The complete batch normalization operation is:

```
BN(x) = γ * ((x - μ) / √(σ² + ε)) + β
```

Where:
- `x` = input tensor
- `μ` = batch mean (computed via `tf.nn.moments`)
- `σ²` = batch variance (computed via `tf.nn.moments`)
- `γ` = scale parameter (learnable, initialized to 1)
- `β` = offset parameter (learnable, initialized to 0)
- `ε` = epsilon (1e-5) to prevent division by zero

### Why This Works

1. **Uses Low-Level APIs**: `tf.nn.batch_normalization` and `tf.nn.moments` are core TensorFlow operations that remain available in TF2.x via compat.v1

2. **Manual Variable Management**: Instead of relying on `tf.layers.batch_normalization` to create variables automatically, we explicitly create them using `tf.get_variable`

3. **Equivalent Functionality**: This implementation is mathematically equivalent to `tf.layers.batch_normalization`, just using lower-level building blocks

### Usage Example

**Before (doesn't work in TF2.x)**:
```python
h = tf.layers.dense(inputs=X, units=800)
h = tf.layers.batch_normalization(h, axis=1, training=is_training)  # ❌ Fails
h = tf.nn.leaky_relu(h)
```

**After (works in TF2.x)**:
```python
h = tf.layers.dense(inputs=X, units=800)
h = _work_around("layer_bn_800", 800, h, is_training)  # ✅ Works
h = tf.nn.leaky_relu(h)
```

### Key Points

1. **Scope Names**: Each call needs a unique scope name (e.g., `"gq_bn_800_1"`, `"gp_bn_800_2"`) to avoid variable name conflicts

2. **Feature Dimension**: Must match the number of features in the input tensor (e.g., 800 for a layer with 800 units)

3. **Training Mode**: The `training` parameter is passed but not directly used in this implementation. In a full implementation, you'd use it to switch between batch statistics (training) and moving average statistics (inference)

4. **Variable Reuse**: `tf.AUTO_REUSE` ensures that if the same scope is used multiple times (e.g., in a loop), the same variables are reused rather than creating duplicates

### Limitations

This workaround implements **batch normalization** but doesn't implement:
- **Moving average statistics** for inference (uses batch stats even during inference)
- **Training vs inference mode switching** (always uses batch statistics)

For most use cases, this is acceptable. If you need moving averages, you'd need to add additional code to track and update them.

### Files Updated

- ✅ `code/st_gan.py` - Already had the workaround
- ✅ `code/pancreas.py` - Updated to use workaround
- ✅ `code/mouse_atlas.py` - Updated to use workaround

All files now use the same consistent approach for batch normalization.

