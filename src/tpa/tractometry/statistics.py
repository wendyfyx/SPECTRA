import logging
from pathlib import Path
from tqdm import tqdm
from typing import List

import numpy as np
import pandas as pd

def aggregate_profiles(metric: str, 
                       bundle: str, 
                       root_dir: str, 
                       subj_list: List[str] | None, 
                       mode: str = '2d'):
    """
    Aggregate tractometry profiles for a given metric, bundle and subject list into a dataframe and save to file.
    Assume the profiles are organized in this format [root_dir]/[metric]/[bundle]/[subject].npz
    
    Args
    ----------
    metric : str
        Metric name, e.g., 'fa'
    bundle : str
        Bundle name, e.g., 'AF_L'
    root_dir : str
        Root directory containing the metric/bundle folders
    subj_list : list
        List of subjects to aggregate profiles for 

    Returns
    -------
    group_average_profile : np.ndarray, shape (ns, nr)
        Group average profile
    df: pd.DataFrame
        A dataframe aggregated bundle profile
    """
    bundle_dir = Path(root_dir) / metric / bundle
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Directory {bundle_dir} does not exist")
    
    print(mode)
    data = []
    subjects = []

    if subj_list is None:
        subj_list = [file.stem for file in bundle_dir.glob("*.npz")]
    
    # Get subjects whose bundle profile exists
    for s in tqdm(subj_list):
        fpath = f"{bundle_dir}/{s}.npz"
        if Path(fpath).is_file():
            data.append(np.load(fpath)[f'profile_{mode}'])
            subjects.append(s)

    # Aggregate into dataframe
    data = np.stack(data)  # shape: (N, ns, nr)
    if mode=='1d':
        data = np.expand_dims(data, axis=2)
    N, ns, nr = data.shape
    flat = data.reshape(N, ns * nr)
    df = pd.DataFrame(flat, columns=[f"s{s}_r{r}" for s in range(ns) for r in range(nr)])
    df.insert(0, "subject", subjects)

    # Ensure numeric columns are float
    for col in df.columns[1:]:
        df[col] = df[col].astype(float)

    logging.info(f"Aggregated {mode} {metric} profiles for {bundle} and {len(df)} subjects with ({ns}, {nr}) grid ")
    return np.mean(data, axis=0), df