# AnnData View and Sparse Matrix Modernization

## Overview

This document describes the modernization of the scgen-reproducibility repository to handle changes in anndata's view and sparse matrix handling between 2018 and modern versions.

## Problem Statement

In modern anndata (post-2018), there are significant differences in how views and sparse matrices are handled:

1. **Views**: Subsetting operations (e.g., `adata[adata.obs["cell_type"] == "CD4T"]`) now create views that reference the parent object, rather than independent copies.
2. **Sparse Matrices**: Accessing `.X.A` on a view may still reference the sparse parent data, leading to unexpected behavior.
3. **Edge Cases**: Nested views and subsets of views can still reference sparse data even after conversion attempts.

## Solution: Utility Shim Functions

Two utility functions have been created in `code/scgen/file_utils.py`:

### 1. `to_dense(adata, copy_if_view=True)`

Converts an AnnData object to dense format, properly handling views and sparse matrices.

**Key Features:**
- Detects if the object is a view using `is_view` attribute or `_parent` reference
- Copies views before conversion to ensure independence
- Converts sparse matrices to dense numpy arrays
- Handles edge cases where X itself might be a view
- Preserves all metadata (obs, var, uns, obsm, varm, obsp, varp, layers)

**Usage:**
```python
from scgen.file_utils import to_dense
view = adata[adata.obs["cell_type"] == "CD4T"]
dense_adata = to_dense(view)
```

### 2. `get_dense_X(adata, copy_if_view=True)`

Convenience function that extracts only the dense X matrix.

**Usage:**
```python
from scgen.file_utils import get_dense_X
view = adata[adata.obs["cell_type"] == "CD4T"]
dense_X = get_dense_X(view)
```

## Implementation Details

### View Detection

The utility function uses multiple strategies to detect views:
1. Checks for `is_view` attribute (anndata >= 0.7.0)
2. Checks for `_parent` reference (alternative detection method)
3. Falls back safely if detection fails

### Sparse Matrix Conversion

When a sparse matrix is detected:
1. If the object is a view, it's copied first
2. The sparse matrix is converted using `.toarray()`
3. A new AnnData object is created with dense X, preserving all metadata

### Edge Case Handling

- **Nested Views**: If X itself is a view (detected via `base` attribute), it's properly copied
- **Already Dense**: If X is already dense but the object was a view, the copied object is returned
- **Safe Defaults**: When in doubt, the function creates a copy to ensure safety

## Files Updated

### Core Utilities
- ✅ `code/scgen/file_utils.py` - Added `to_dense()` and `get_dense_X()` functions
- ✅ `code/scgen/__init__.py` - Exported new utility functions

### Main Code Updates
- ✅ `code/scgen/models/util.py` - Updated:
  - `training_data_provider()` - Uses `get_dense_X()` instead of `.X.A`
  - `shuffle_data()` - Uses `get_dense_X()` for view-safe access
  - `balancer()` - Uses `get_dense_X()` for sparse matrix handling
  - `batch_removal()` - Uses `get_dense_X()` for latent space conversion
  - `visualize_trained_network_results()` - Updated all three network type branches

- ✅ `code/data_reader.py` - Updated:
  - `training_data_provider()` - Uses `get_dense_X()` instead of `.X.A`

## Migration Status

✅ **All files have been migrated!** All instances of `.X.A` have been replaced with the new utility functions.

### Files Migrated
- ✅ `code/scgen/models/_vae.py` - All instances updated
- ✅ `code/scgen/models/_vae_keras.py` - All instances updated
- ✅ `code/scgen/models/_cvae.py` - All instances updated
- ✅ `code/vec_arith.py` - All instances updated
- ✅ `code/vec_arith_pca.py` - All instances updated
- ✅ `code/st_gan.py` - All instances updated
- ✅ `code/train_scGen.py` - All instances updated
- ✅ `code/scgen/plotting.py` - All instances updated

## Migration Strategy

For remaining files, the recommended approach is:

1. **Import the utility:**
   ```python
   from scgen.file_utils import get_dense_X, to_dense
   ```

2. **Replace `.X.A` with `get_dense_X(adata)`:**
   ```python
   # Old
   dense_data = adata.X.A
   
   # New
   dense_data = get_dense_X(adata)
   ```

3. **Replace in-place conversions:**
   ```python
   # Old
   if sparse.issparse(adata.X):
       adata.X = adata.X.A
   
   # New
   adata = to_dense(adata)
   ```

## Testing Recommendations

1. **View Handling**: Test with views created from subsetting operations
2. **Sparse Matrices**: Test with both sparse and dense input data
3. **Nested Views**: Test views of views to ensure proper copying
4. **Memory**: Verify that copies are made when necessary but not unnecessarily
5. **Backward Compatibility**: Ensure existing code continues to work

## Edge Cases Addressed

1. ✅ Views created from subsetting operations
2. ✅ Sparse matrices in views
3. ✅ Nested views (views of views)
4. ✅ X matrix that is itself a view
5. ✅ Already-dense data in views
6. ✅ Mixed sparse/dense scenarios

## Notes

- The utility functions are designed to be safe by default (copying when in doubt)
- Performance impact is minimal as copies are only made when necessary
- The functions handle both modern and legacy anndata versions
- All metadata is preserved during conversion
