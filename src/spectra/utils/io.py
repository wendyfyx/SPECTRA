import os
import logging
import json
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
from dipy.io.image import load_nifti, save_nifti
from dipy.io.streamline import load_trk, save_trk
from dipy.io.stateful_tractogram import Space, StatefulTractogram

from spectra.utils.utils import file_exists

ListItem = Union[str, int, float]

def _json_default(obj: Any) -> Any:
    """Handle non-serializable types for JSON encoding."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def load_list(filepath, is_str=False):
    """
    Load data from a .txt or .npy file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the file.
    is_str : bool, default=False
        If True, force loading as list of strings.
    
    Returns
    -------
    np.ndarray or list[str]
        - np.ndarray for numeric data
        - list[str] for string data
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # --- Force string reading ---
    if is_str:
        with open(filepath, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    # --- Normal behavior ---
    if filepath.suffix == ".npy":
        data = np.load(filepath, allow_pickle=True)
        return np.atleast_1d(data)  # ensure 1D if scalar or 0D array

    elif filepath.suffix == ".txt":
        try:
            data = np.loadtxt(filepath)
            return np.atleast_1d(data)
        except ValueError:
            # fallback if text not numeric
            with open(filepath, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines

    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")


def save_list_to_txt(file_path: str, data: List[ListItem], force: bool = False) -> None:
    """
    Save a list of strings, ints, or floats to a text file, one element per line.
    
    Args:
        file_path: Path to the text file to write.
        data: List of elements (str, int, float) to write.
    """
    if file_exists(file_path) and not force:
        logging.info(f"<{file_path}> already exists")
        return

    if len(data)==0:
        logging.info("Empty data")
        return
    if not isinstance(data, (list, tuple, np.ndarray)):
        data = [data]
    if isinstance(data, np.ndarray):
        np.savetxt(file_path, data)
    else:
        with open(file_path, 'w') as f:
            for item in data:
                f.write(f"{item}\n")


def load_json(file_path: str) -> Any:
    """
    Read Python data from a JSON file.
    
    Args:
        file_path: Path to the JSON file to read.
        
    Returns:
        Python object (dict, list, etc.) from the JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(file_path: str, data: Any, indent: int = 4, force: bool = False) -> None:
    """
    Write Python data (dict, list, etc.) to a JSON file.
    
    Args:
        file_path: Path to the JSON file to write.
        data: Python data (dict, list, etc.) to write.
        indent: Number of spaces for indentation (default 4).
    """
    if file_exists(file_path) and not force:
        logging.info(f"<{file_path}> already exists")
    else:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent, default=_json_default)


def load_jsonl(file_path: str) -> list[Any]:
    """
    Read Python data from a JSONL file (one JSON object per line).
    
    Args:
        file_path: Path to the JSONL file to read.
        
    Returns:
        List of Python objects from the JSONL file.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(file_path: str, data: list[Any], force: bool = False) -> None:
    """
    Write a list of Python objects to a JSONL file (one JSON object per line).
    
    Args:
        file_path: Path to the JSONL file to write.
        data: List of Python objects to write.
        force: If True, overwrite existing file.
    """
    if file_exists(file_path) and not force:
        logging.info(f"<{file_path}> already exists")
    else:
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item, default=_json_default) + '\n')


def save_csv(file_path: str, df: pd.DataFrame, force: bool = False):
    if file_exists(file_path) and not force:
        logging.info(f"<{file_path}> already exists")
    else:
        df.to_csv(file_path, index=False)


def save_fig(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    logging.debug(f"Saved figure: {path}")


def save_npy(arr: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logging.debug(f"Saved array: {path}")