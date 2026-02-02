import os

import numpy as np
from scipy import sparse
import anndata


def _invalid_pairwise_keys(container, n_obs, keys):
    invalid_keys = []
    for key in keys:
        value = container.get(key)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is None or shape != (n_obs, n_obs):
            invalid_keys.append(key)
    return invalid_keys


def _safe_delete_mapping_keys(container, keys):
    for key in keys:
        try:
            del container[key]
        except Exception:
            pass


def _drop_invalid_obsp(adata):
    """
    Remove obsp entries that do not match (n_obs, n_obs).

    Some legacy datasets include stale neighbor graphs (e.g., 'distances')
    that no longer align with the current obs dimension. These entries cause
    AnnData copy/validation to fail in modern anndata.
    """
    # Avoid `hasattr(adata, "obsp")` because AnnData's descriptor can raise
    # during access if obsp entries are invalid. We want to handle that case.
    target = getattr(adata, "_parent", None) or adata
    try:
        obsp = target.obsp
        items = list(obsp.items())
    except Exception:
        # Fall back to the private store to remove invalid entries without
        # triggering validation in the public accessor.
        obsp = getattr(target, "_obsp", None)
        if not hasattr(obsp, "items"):
            return []
        items = list(obsp.items())
    n_obs = target.n_obs
    container = dict(items)
    invalid_keys = _invalid_pairwise_keys(container, n_obs, container.keys())
    for key in invalid_keys:
        try:
            del target._obsp[key]
            continue
        except Exception:
            pass
    _safe_delete_mapping_keys(getattr(target, "obsp", {}), invalid_keys)
    return invalid_keys


def _drop_invalid_uns_neighbors(adata):
    """
    Remove legacy .uns['neighbors'] entries that do not match (n_obs, n_obs).
    """
    target = getattr(adata, "_parent", None) or adata
    try:
        neighbors = target.uns.get("neighbors")
    except Exception:
        return []
    if not isinstance(neighbors, dict):
        return []

    n_obs = target.n_obs
    invalid_keys = _invalid_pairwise_keys(neighbors, n_obs, ("distances", "connectivities"))

    for key in invalid_keys:
        _safe_delete_mapping_keys(neighbors, [key])

    if len(neighbors) == 0:
        try:
            target.uns.pop("neighbors", None)
        except Exception:
            pass

    return invalid_keys


def _is_view(adata):
    try:
        if hasattr(adata, "is_view"):
            return adata.is_view
        if hasattr(adata, "_parent"):
            return adata._parent is not None
    except (AttributeError, RuntimeError):
        return False
    return False


def _dense_array_from_X(X, *, copy_if_view=True, is_view=False):
    if sparse.issparse(X):
        return X.toarray()
    if copy_if_view and is_view:
        return np.array(X, copy=True)
    try:
        if hasattr(X, "base") and X.base is not None:
            return np.array(X, copy=True)
    except Exception:
        return np.array(X, copy=True)
    if isinstance(X, np.ndarray):
        return X
    return np.array(X, copy=True)


def to_dense(adata, copy_if_view=True):
    """
    Converts an AnnData object to dense format, handling views and sparse matrices.
    
    In modern anndata (post-2018), subsetting operations create views that reference
    the parent object. This function ensures that:
    1. If the object is a view, it is copied first (if copy_if_view=True)
    2. Sparse matrices are converted to dense numpy arrays
    3. Edge cases with nested views are handled
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to convert to dense format
    copy_if_view : bool, optional
        If True (default), copy the object if it's a view before converting.
        If False, attempt to convert in-place (may fail for views).
    
    Returns
    -------
    anndata.AnnData
        A new AnnData object with dense X matrix. If the input was not a view
        and X was already dense, returns the same object (unless copy was needed).
    
    Example
    -------
    >>> import anndata
    >>> import scgen
    >>> adata = anndata.read("data.h5ad")
    >>> # Create a view (subset)
    >>> view = adata[adata.obs["cell_type"] == "CD4T"]
    >>> # Convert to dense, handling the view properly
    >>> dense_adata = scgen.file_utils.to_dense(view)
    """
    is_view = _is_view(adata)
    
    # If it's a view and we should copy, do so first
    # This ensures we have an independent object before converting
    if is_view and copy_if_view:
        _drop_invalid_obsp(adata)
        adata = adata.copy()
    
    # Convert sparse matrix to dense if needed
    if sparse.issparse(adata.X):
        dense_X = _dense_array_from_X(adata.X, copy_if_view=copy_if_view, is_view=is_view)
        # Create new AnnData with dense X, preserving all metadata
        # Use copy() method which handles all attributes properly
        _drop_invalid_obsp(adata)
        result = adata.copy()
        result.X = dense_X
        return result
    else:
        # Already dense
        # If we copied because it was a view, return the copy
        # Otherwise, if X might still reference parent data, copy to be safe
        if is_view and copy_if_view:
            return adata
        
        # Check if X itself is a view (e.g., from slicing a numpy array)
        dense_X = _dense_array_from_X(adata.X, copy_if_view=copy_if_view, is_view=is_view)
        if dense_X is not adata.X:
            _drop_invalid_obsp(adata)
            result = adata.copy()
            result.X = dense_X
            return result
        
        # If we get here, X is already dense and not a view
        # Return as-is (it's safe)
        return adata


def get_dense_X(adata, copy_if_view=True):
    """
    Extracts the dense X matrix from an AnnData object, handling views and sparse matrices.
    
    This is a convenience function for cases where you only need the X array,
    not the full AnnData object.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object
    copy_if_view : bool, optional
        If True (default), copy the object if it's a view before converting.
    
    Returns
    -------
    numpy.ndarray
        Dense numpy array of the X matrix
    
    Example
    -------
    >>> import anndata
    >>> import scgen
    >>> adata = anndata.read("data.h5ad")
    >>> view = adata[adata.obs["cell_type"] == "CD4T"]
    >>> dense_X = scgen.file_utils.get_dense_X(view)
    """
    return _dense_array_from_X(
        adata.X,
        copy_if_view=copy_if_view,
        is_view=_is_view(adata),
    )


def ensure_dir_for_file(file_path):
    """
    Ensures the directory for a file path exists, creating it if necessary.

    Parameters
    ----------
    file_path : str
        Path to the file (can be absolute or relative)

    Returns
    -------
    str
        The original file_path (for chaining)

    Example
    -------
    >>> from scgen.file_utils import ensure_dir_for_file
    >>> file_path = ensure_dir_for_file("../data/reconstructed/VecArithm/file.h5ad")
    >>> adata.write(file_path)
    """
    file_dir = os.path.dirname(file_path)
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    return file_path


def ensure_dir(dir_path):
    """
    Ensures a directory exists, creating it if necessary.

    Parameters
    ----------
    dir_path : str
        Path to the directory

    Returns
    -------
    str
        The original dir_path (for chaining)

    Example
    -------
    >>> from scgen.file_utils import ensure_dir
    >>> ensure_dir("../results/Figures/")
    """
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return dir_path


def should_skip_reconstruction(file_path, overwrite=False):
    """
    Returns True if the reconstruction output already exists and should be reused.

    Parameters
    ----------
    file_path : str
        Path to the reconstruction output file
    overwrite : bool, optional
        If True, skip the check and allow regeneration
    """
    env_overwrite = os.environ.get("SCGEN_OVERWRITE", "").strip().lower() in ("1", "true", "yes")
    if overwrite or env_overwrite:
        return False
    if file_path and os.path.isfile(file_path):
        print(f"Reconstruction output already exists, skipping: {file_path}")
        return True
    return False