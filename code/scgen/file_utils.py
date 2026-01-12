import os


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
