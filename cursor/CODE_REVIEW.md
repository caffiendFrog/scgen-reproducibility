# Code Review: Shape Mismatch Fix & Library Compatibility Updates

## Executive Summary

**Overall Assessment**: ✅ **APPROVED with Minor Improvements Recommended**

The proposed changes correctly address the shape mismatch bug and add necessary validation. The fixes are sound, but there are opportunities for improvement in error handling, code organization, and edge case coverage.

---

## 1. `code/train_scGen.py` - `reconstruct_whole_data()` Function

### ✅ Strengths

1. **Correct Core Fix**: Using `net_train_data.X.shape[1]` instead of `train.X.shape[1]` is the right solution
2. **Good Logging**: `log_shape()` helper provides useful debugging information
3. **Defensive Programming**: Multiple validation checks catch issues early
4. **Gene Mapping Logic**: Handles the case where prediction uses subset of genes

### ⚠️ Issues & Recommendations

#### **CRITICAL: Redundant Assertion**

**Lines 123-124**:
```python
assert network.x_dim == net_train_data.X.shape[1], \
    f"Model x_dim ({network.x_dim}) != net_train_data genes ({net_train_data.X.shape[1]})"
```

**Issue**: This assertion is redundant with the more informative `ValueError` check at lines 135-140. Assertions can be disabled with `-O` flag, making them unreliable for production code.

**Recommendation**: Remove the assertion and keep only the `ValueError` check:
```python
# Remove lines 123-124, keep lines 135-140
```

#### **MEDIUM: Gene Mapping Edge Cases**

**Lines 158-189**: The gene mapping logic has potential issues:

1. **Missing validation for empty gene sets**:
```python
if len(net_train_data.var_names) == 0:
    raise ValueError("net_train_data has no genes")
```

2. **Inefficient dictionary creation**: Creating dictionaries for every cell type in a loop could be optimized:
```python
# Consider caching gene mappings if called multiple times
```

3. **Silent gene loss**: If `mapped_count != pred.shape[1]`, we raise an error, but we should also warn about which genes couldn't be mapped:
```python
unmapped_genes = set(net_train_data.var_names) - set(full_gene_idx.keys())
if unmapped_genes:
    print(f"WARNING: {len(unmapped_genes)} genes could not be mapped: {list(unmapped_genes)[:5]}...")
```

#### **LOW: Error Handling**

**Line 213**: File I/O operation lacks error handling:
```python
all_data.write_h5ad(f"../data/reconstructed/scGen/{data_name}.h5ad")
```

**Recommendation**:
```python
try:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_data.write_h5ad(output_path)
except (IOError, OSError) as e:
    raise RuntimeError(f"Failed to write reconstruction output to {output_path}: {e}")
```

#### **LOW: Code Organization**

The `reconstruct_whole_data()` function is quite long (140+ lines). Consider extracting:
- Gene mapping logic into a separate function
- Data loading logic into a helper

---

## 2. `code/scgen/models/_vae.py` - `to_latent()` Method

### ✅ Strengths

1. **Clear Error Message**: The ValueError provides actionable information
2. **Early Detection**: Catches shape mismatches before TensorFlow execution

### ⚠️ Issues & Recommendations

#### **MEDIUM: Missing Edge Case Checks**

**Current code**:
```python
if data.shape[1] != self.x_dim:
    raise ValueError(...)
```

**Issues**:
1. Doesn't check if `data` is empty (0 rows)
2. Doesn't validate that `data` is 2D
3. Doesn't handle 1D array edge case

**Recommendation**:
```python
# Validate input shape
if not isinstance(data, np.ndarray):
    raise TypeError(f"Expected numpy array, got {type(data)}")
if data.ndim != 2:
    raise ValueError(f"Expected 2D array, got {data.ndim}D array with shape {data.shape}")
if data.shape[0] == 0:
    raise ValueError(f"Input data has 0 samples (rows)")
if data.shape[1] != self.x_dim:
    raise ValueError(
        f"Input data has {data.shape[1]} features, but model expects {self.x_dim} features. "
        f"Shape: {data.shape}, Expected: [n_cells, {self.x_dim}]. "
        f"This usually indicates a mismatch between training and inference data preprocessing."
    )
```

#### **LOW: Documentation Update**

The docstring mentions `data.X` but the function actually takes a numpy array directly. Update docstring:
```python
# Parameters
    data: numpy nd-array
        Numpy nd-array to be mapped to latent space. Must be in shape [n_obs, n_vars].
        Note: This function expects a numpy array, not an AnnData object.
```

---

## 3. `code/test_reconstruct_shape.py` - Test File

### ✅ Strengths

1. **Clear Structure**: Well-organized test with good output
2. **Covers Main Cases**: Tests correct shapes, incorrect shapes, sparse matrices

### ⚠️ Issues & Recommendations

#### **MEDIUM: Test Framework**

**Current**: Uses manual assertions and print statements

**Recommendation**: Use pytest for better test reporting:
```python
import pytest

def test_model_initialization():
    """Test that model initialization matches data dimensions"""
    adata = create_test_data()
    network = scgen.VAEArith(x_dimension=adata.X.shape[1], z_dimension=10)
    assert network.x_dim == adata.X.shape[1]

def test_to_latent_correct_shape():
    """Test to_latent with correct shape"""
    # ... test code

def test_to_latent_incorrect_shape():
    """Test to_latent raises ValueError for incorrect shape"""
    with pytest.raises(ValueError, match="Input data has.*features"):
        # ... test code
```

#### **MEDIUM: Missing Test Cases**

1. **Gene mapping logic**: No test for the gene space mapping in `reconstruct_whole_data()`
2. **Empty data**: No test for empty arrays
3. **1D array edge case**: No test for 1D input
4. **Sparse matrix in to_latent**: Test doesn't actually call `to_latent()` with sparse data

**Recommendation**: Add tests:
```python
def test_gene_mapping():
    """Test gene space mapping when prediction has fewer genes"""
    # Create test scenario where pred has subset of genes
    # Verify mapping works correctly

def test_empty_data():
    """Test that empty data raises appropriate error"""
    network = scgen.VAEArith(x_dimension=50, z_dimension=10)
    with pytest.raises(ValueError, match="0 samples"):
        network.to_latent(np.array([]).reshape(0, 50))
```

#### **LOW: Test Data Creation**

Consider extracting test data creation into a fixture:
```python
@pytest.fixture
def test_adata():
    """Create standard test AnnData object"""
    n_cells, n_genes = 100, 50
    X = np.random.RandomState(42).rand(n_cells, n_genes)
    obs = {
        "cell_type": ["A"] * 50 + ["B"] * 50,
        "condition": (["control"] * 25 + ["stimulated"] * 25) * 2
    }
    var = {"var_names": [f"Gene_{i}" for i in range(n_genes)]}
    return anndata.AnnData(X, obs=obs, var=var)
```

---

## 4. `code/scgen/models/util.py` - Typo Fix

### ✅ Assessment

**Line 131**: Fix from `train_t_x.append()` to `train_t_X.append()` is correct.

**No issues found** - Simple typo correction.

---

## 5. `code/st_gan.py` - TensorFlow Import Fix

### ✅ Assessment

**Line 8**: Fix from `import tf.compat.v1 as tf` to `import tensorflow.compat.v1 as tf` is correct.

**No issues found** - Simple import correction.

---

## 6. `log_shape()` Helper Function

### ⚠️ Issues & Recommendations

#### **LOW: Type Checking**

**Current code** (lines 59-69):
```python
def log_shape(name, obj):
    """Helper to log shapes for debugging"""
    if hasattr(obj, 'shape'):
        print(f"[SHAPE] {name}: {obj.shape}")
    elif hasattr(obj, 'X'):
        # ...
```

**Issues**:
1. Uses `hasattr()` which can be slow and catch unexpected attributes
2. Doesn't handle all AnnData edge cases (e.g., when `X` is None)

**Recommendation**:
```python
def log_shape(name, obj):
    """Helper to log shapes for debugging"""
    import anndata
    
    if isinstance(obj, anndata.AnnData):
        x_shape = obj.X.shape if obj.X is not None else "None"
        print(f"[SHAPE] {name}: {obj.shape} (X: {x_shape})")
    elif isinstance(obj, np.ndarray):
        print(f"[SHAPE] {name}: {obj.shape}")
    elif hasattr(obj, 'shape'):
        print(f"[SHAPE] {name}: {obj.shape}")
    else:
        print(f"[SHAPE] {name}: {type(obj)} (no shape attribute)")
```

#### **LOW: Logging Level**

Consider using Python's `logging` module instead of `print()`:
```python
import logging
logger = logging.getLogger(__name__)

def log_shape(name, obj):
    """Helper to log shapes for debugging"""
    # ... shape detection logic ...
    logger.debug(f"[SHAPE] {name}: {shape_info}")
```

---

## Summary of Required Changes

### 🔴 Critical (Must Fix)
1. **Remove redundant assertion** in `reconstruct_whole_data()` (lines 123-124)

### 🟡 Medium Priority (Should Fix)
1. **Add edge case validation** to `to_latent()` (empty arrays, 1D arrays, type checking)
2. **Improve gene mapping** error messages (show which genes couldn't be mapped)
3. **Add missing test cases** (gene mapping, empty data, 1D arrays)
4. **Consider using pytest** for test framework

### 🟢 Low Priority (Nice to Have)
1. **Add error handling** for file I/O operations
2. **Refactor long functions** (extract gene mapping logic)
3. **Use logging module** instead of print statements
4. **Improve type checking** in `log_shape()`

---

## Testing Recommendations

1. **Run existing tests**: Verify no regressions
2. **Test with real data**: Run `reconstruct_whole_data()` with actual h5ad files
3. **Test edge cases**: Empty data, single cell, single gene
4. **Performance test**: Verify gene mapping doesn't slow down significantly

---

## Code Quality Metrics

- **Maintainability**: ⭐⭐⭐⭐ (Good - clear intent, good comments)
- **Robustness**: ⭐⭐⭐ (Medium - needs more edge case handling)
- **Test Coverage**: ⭐⭐⭐ (Medium - covers main cases, missing edge cases)
- **Documentation**: ⭐⭐⭐⭐ (Good - clear comments and error messages)

---

## Final Verdict

✅ **APPROVED** - The changes correctly fix the shape mismatch bug and add valuable validation. The recommended improvements would enhance robustness and maintainability but are not blockers.

**Recommended Action**: Address the critical issue (redundant assertion) and at least the medium-priority edge case validations before merging.

