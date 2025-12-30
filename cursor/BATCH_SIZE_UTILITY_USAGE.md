# Batch Size Utility Function Usage Guide

## Overview

The `determine_batch_size()` function in `utils.py` intelligently calculates optimal batch sizes based on:
- Input dimension (X_dim) - number of features/genes
- Dataset size - total number of samples
- Safe/aggressive batch size limits
- Model type (VAE, CVAE, ST-GAN, etc.)
- Available GPU memory

## Function Signature

```python
def determine_batch_size(x_dim, dataset_size, safe_batch_size, aggressive_batch_size, 
                         model_type='vae', gpu_memory_gb=24, use_aggressive=False):
```

## Parameters

- **x_dim** (int): Input dimension (number of features/genes)
- **dataset_size** (int): Total number of samples in dataset
- **safe_batch_size** (int): Conservative maximum batch size
- **aggressive_batch_size** (int): Aggressive maximum batch size
- **model_type** (str): 'vae', 'cvae', 'stgan', 'mouse_atlas', or 'pancreas' (default: 'vae')
- **gpu_memory_gb** (int): Available GPU memory in GB (default: 24 for g5.4xlarge)
- **use_aggressive** (bool): Use aggressive limits if True (default: False)

## Usage Examples

### Example 1: VAE Training

```python
from utils import determine_batch_size, get_gpu_memory_gb

# Get dataset info
train_data = sc.read("../data/train_pbmc.h5ad")
x_dim = train_data.X.shape[1]  # Number of genes
dataset_size = train_data.shape[0]  # Number of cells

# Get GPU memory (optional, falls back to 24GB if detection fails)
gpu_memory = get_gpu_memory_gb() or 24

# Determine batch size
batch_size = determine_batch_size(
    x_dim=x_dim,
    dataset_size=dataset_size,
    safe_batch_size=256,
    aggressive_batch_size=512,
    model_type='vae',
    gpu_memory_gb=gpu_memory,
    use_aggressive=False  # Use safe mode
)

print(f"Using batch size: {batch_size}")
network.train(train_data, batch_size=batch_size, ...)
```

### Example 2: ST-GAN Training

```python
from utils import determine_batch_size

# ST-GAN typically has larger batch sizes
batch_size = determine_batch_size(
    x_dim=gex_size,  # Gene expression size
    dataset_size=len(train_real_stim),
    safe_batch_size=2048,
    aggressive_batch_size=4096,
    model_type='stgan',
    gpu_memory_gb=24,
    use_aggressive=True  # Use aggressive mode for faster training
)

print(f"ST-GAN batch size: {batch_size}")
```

### Example 3: CVAE Training

```python
from utils import determine_batch_size

batch_size = determine_batch_size(
    x_dim=train.X.shape[1],
    dataset_size=train.shape[0],
    safe_batch_size=256,
    aggressive_batch_size=512,
    model_type='cvae',
    use_aggressive=False
)

network.train(train, batch_size=batch_size, ...)
```

### Example 4: Integration in train_scGen.py

```python
from utils import determine_batch_size

def test_train_whole_data_one_celltype_out(data_name="pbmc",
                                           z_dim=50,
                                           alpha=0.1,
                                           n_epochs=1000,
                                           batch_size=None,  # Auto-determine if None
                                           dropout_rate=0.25,
                                           learning_rate=0.001,
                                           condition_key="condition",
                                           cell_type_to_train=None):
    # ... load data ...
    
    for cell_type in train.obs[cell_type_key].unique().tolist():
        net_train_data = train[~((train.obs[cell_type_key] == cell_type) & 
                                 (train.obs[condition_key] == stim_key))]
        
        # Auto-determine batch size if not provided
        if batch_size is None:
            batch_size = determine_batch_size(
                x_dim=net_train_data.X.shape[1],
                dataset_size=net_train_data.shape[0],
                safe_batch_size=256,
                aggressive_batch_size=512,
                model_type='vae'
            )
            print(f"Auto-determined batch size for {cell_type}: {batch_size}")
        
        network = scgen.VAEArith(...)
        network.train(net_train_data, batch_size=batch_size, ...)
```

## How It Works

1. **Model Type Scaling**: Different models have different memory requirements:
   - VAE: Baseline (factor 1.0)
   - CVAE: 0.9x (slightly smaller network)
   - ST-GAN: 1.5x (2 generators + 2 discriminators)

2. **GPU Memory Scaling**: Scales batch size based on available GPU memory relative to 24GB baseline

3. **Input Dimension Scaling**: Larger X_dim (more genes) = less memory per sample = smaller batch size

4. **Dataset Size Limit**: Never exceeds the actual dataset size

5. **Power of 2 Rounding**: Rounds down to nearest power of 2 for GPU efficiency

## Recommended Safe/Aggressive Values for g5.4xlarge

| Model Type | Safe Max | Aggressive Max |
|------------|----------|----------------|
| VAE/CVAE | 256 | 512 |
| ST-GAN | 2048 | 4096 |
| Mouse Atlas/Pancreas | 256 | 512 |

## Integration Points

You can integrate this function in:

1. **st_gan.py** (line 54): Replace `batch_size = 512` with auto-determination
2. **train_scGen.py** (lines 14, 228, 263): Add batch_size parameter with auto-determination
3. **train_cvae.py** (line 12): Add explicit batch_size parameter
4. **mouse_atlas.py** (line 25): Replace `batch_size = 32` with auto-determination
5. **pancreas.py** (line 27): Replace `batch_size = 32` with auto-determination

## Benefits

1. **Automatic Optimization**: No manual tuning needed
2. **GPU-Aware**: Adapts to available GPU memory
3. **Dataset-Aware**: Never exceeds dataset size
4. **Model-Aware**: Accounts for different model memory requirements
5. **Dimension-Aware**: Adjusts for input feature count

## Testing

```python
# Test with different scenarios
test_cases = [
    (3000, 10000, 256, 512, 'vae'),      # Typical VAE
    (5000, 50000, 2048, 4096, 'stgan'), # Large ST-GAN
    (2000, 5000, 256, 512, 'cvae'),     # Small CVAE
]

for x_dim, dataset_size, safe, agg, model_type in test_cases:
    batch_size = determine_batch_size(x_dim, dataset_size, safe, agg, model_type)
    print(f"X_dim={x_dim}, Dataset={dataset_size}, Model={model_type} -> Batch={batch_size}")
```

