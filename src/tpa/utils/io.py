import os
import logging
import json
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import pandas as pd

import nibabel as nib
from dipy.io.image import load_nifti, save_nifti
from dipy.io.streamline import load_trk, save_trk
from dipy.io.stateful_tractogram import Space, StatefulTractogram

from tpa.utils.utils import file_exists

ListItem = Union[str, int, float]


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
            json.dump(data, f, indent=indent)


def save_csv(file_path: str, df: pd.DataFrame, indent: int = 4, force: bool = False):
    if file_exists(file_path) and not force:
        logging.info(f"<{file_path}> already exists")
    else:
        df.to_csv(file_path, index=False)


def load_streamlines(trk_file):
    '''Helper function for loading streamlines'''
    tractogram = load_trk(trk_file, reference='same', bbox_valid_check=False)
    return tractogram.streamlines, tractogram.affine


def save_streamlines(streamlines, out_file, ref_file):
    '''Save streamlines to file given a reference file'''
    sft = StatefulTractogram(streamlines, ref_file, Space.RASMM)
    sft.remove_invalid_streamlines()
    save_trk(sft, out_file, bbox_valid_check=False)
    logging.info(f'Saved {len(streamlines)} streamlines to {out_file}')