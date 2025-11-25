import sys
import logging
from pathlib import Path
from importlib.resources import files

import numpy as np
from scipy.ndimage import label

def setup_logging():
    """Setup logging using default config"""
    logging.basicConfig(
        stream=sys.stdout,
        level="INFO",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        encoding='utf-8',
        force=True
    )

def get_t1_mni_path():
    """Get filename of the T1 MNI template"""
    return str(files("buan_weighted_mean") / "assets" / "mni_icbm152_t1_tal_nlin_asym_09a_masked.nii.gz")

def file_exists(file_path: str) -> bool:
    """
    Check if a file exists at the given path.
    
    Args:
        file_path (str): Path to the file.
    
    Returns:
        bool: True if the file exists and is a file, False otherwise.
    """
    path = Path(file_path)
    return path.is_file()

def directory_exists(dir_path: str) -> bool:
    """
    Check if a directory exists at the given path.
    
    Args:
        dir_path (str): Path to the directory.
    
    Returns:
        bool: True if the directory exists, False otherwise.
    """
    path = Path(dir_path)
    return path.is_dir()

def make_directory(dir_path: str):
    """
    Create a directory, including parent directories (like mkdir -p).
    
    Args:
        dir_path (str): Path to the directory to create.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)

def get_rng(rng=None):
    '''Get a random generator'''
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        rng = rng
    else:
        rng = np.random.default_rng(0)
    return rng

def random_select(data, n_sample=1, rng=0):
    '''
        Randomly select samples from data, returns index
        Example usage: data[random_select(data)]
    '''
    rng = get_rng(rng)
    return rng.choice(len(data), size=n_sample, replace=False)

def clean_mask(mask, min_component_size=8):
    '''Remove small components from mask'''
    labeled, n_components = label(mask)
    clean = np.zeros_like(mask, dtype=bool)
    for comp_id in range(1, n_components + 1):
        comp = (labeled == comp_id)
        if comp.sum() >= min_component_size:
            clean[comp] = True
    return clean