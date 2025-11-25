import logging
import warnings
from pathlib import Path
from tqdm import tqdm
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from tpa.utils.utils import clean_mask

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def aggregate_profiles(metric: str, 
                       bundle: str, 
                       root_dir: str, 
                       subj_list: List[str] | None, 
                       mode: str = '2d', 
                       min_prop: float = 0.3, 
                       disable_tqdm=True):
    """
    Aggregate tractometry profiles for a given metric, bundle and subject list into a dataframe and save to file.
    Assume the profiles are organized in this format [root_dir]/[metric]/[bundle]/[subject].npz
    
    Parameters
    ----------
    metric : str
        Metric name, e.g., 'fa'
    bundle : str
        Bundle name, e.g., 'AF_L'
    root_dir : str
        Root directory containing the metric/bundle folders
    subj_list : list
        List of subjects to aggregate profiles for 
    mode : str
        Creating 1D along-tract or 2D tractometry profile
    min_prop : float
        Minimum proportion of subjects to include in mask

    Returns
    -------
    group_avg : np.ndarray, shape (ns, nr)
        Group average profile
    mask : np.ndarray, shape (ns, nr)
        Profile mask (grid cells with insufficient subjects are masked)
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
    for s in tqdm(subj_list, desc="Aggregating profiles across subjects", 
                  disable=disable_tqdm):
        fpath = f"{bundle_dir}/{s}.npz"
        if Path(fpath).is_file():
            data.append(np.load(fpath)[f'profile_{mode}'])
            subjects.append(s)

    # Aggregate across subjects, create group average profile and mask
    data = np.stack(data)  # shape: (N, ns, nr)
    if mode=='1d':
        data = np.expand_dims(data, axis=2)
    mask = clean_mask((data!=0).mean(axis=0) >= min_prop)
    group_avg = data.mean(axis=0)
    group_avg[~mask] = 0

    # Flatten into dataframe
    n_subj, ns, nr = data.shape
    s_idx, r_idx = np.where(mask)
    col_names = [f'bin_s{s}_r{r}' for s, r in zip(s_idx, r_idx)]
    df = pd.DataFrame(columns=col_names)
    for col, (s, r) in zip(col_names, zip(s_idx, r_idx)):
        df[col] = data[:, s, r]
    df.insert(0, "subject", subjects)

    # flat = data.reshape(N, ns * nr)
    # df = pd.DataFrame(flat, columns=[f"profile_s{s}_r{r}" for s in range(ns) for r in range(nr)])
    # df.insert(0, "subject", subjects)

    # Ensure numeric columns are float
    for col in df.columns[1:]:
        df[col] = df[col].astype(float)

    logging.info(f"Aggregated {mode} {metric} profiles for {bundle} and {len(df)} subjects with ({ns}, {nr}) grid ")

    return group_avg, mask, df

def run_lmm(df, 
            target_col='dx', 
            target_ref_group='control', 
            subj_col='subject',
            site_col="site",
            covar_cols=['age','sex'],
            cont_cols=['age'],
            disable_tqdm=True, 
            min_subjs=10):
    '''
    Run linear mixed model on BUAN profiles. Input dataframe must contain 
    profile_ columns (from aggregate_profile) and other covariates
    Specify `cont_cols` to apply standard scaling
    
    Parameters
    ----------
    df (pd.DataFrame): Input dataframe must have one row per subject, with tract profiles 
                        created from `aggregate_profiles` merged with metadata, and with 
                        informatin on covariates and site.
    target_col (str): Target column, could be categorical or continuous
    target_ref_group (str): If `target_col` is categorical, a reference group could be specified
    site_col (str): Column specifying site to be modeled as random effect if available
    covar_cols (List[str]): List of covariate columns to be modeled as fixed effects
    cont_cols (List[str]): List of continuous columns from either `target_col` or `covar_cols` to be 
                            standardized before running model. The metric column is always standardized.

    Returns
    -------
    pd.DataFrame: A dataframe with statistics for each segment and variable modeled

    '''
    df = df.dropna(subset=[target_col] + covar_cols)
    nsubjs = len(df)

    # Convert to long format for LMM
    df = df.melt(
            id_vars=[col for col in df.columns if "bin_" not in col],
            var_name="bin",
            value_name="metric"
        )

    # Define LMM criteria
    if target_ref_group is not None:
        target_var = f"C({target_col}, Treatment(reference='{target_ref_group}'))"
    else:
        target_var = target_col
    criteria = f"metric ~ {target_var}"
    if len(covar_cols) > 0:
        for covar in covar_cols:
            criteria += f" + {covar}"
    logging.info(f"Running LMM on {nsubjs} subjects with formula <{criteria}>")

    df_lmm = []
    residuals = []
    # Iterate over segments
    for bin in tqdm(df['bin'].unique(), desc="Bin", disable=disable_tqdm):
        df_seg = df.loc[df['bin']==bin]
        subjs = df_seg[subj_col].values
        n_subj = np.count_nonzero(df_seg['metric'])
        if  n_subj < min_subjs:
            continue

        # Standardize continuous columns to mean=0 and std=1
        df_seg.loc[:, 'metric'] = StandardScaler().fit_transform(df_seg[['metric']])
        if len(cont_cols) > 0:
            for c in cont_cols:
                df_seg.loc[:, c] = StandardScaler().fit_transform(df_seg[[c]])

        # Run LMM
        mdf = sm.MixedLM.from_formula(criteria, 
                                      groups=np.ones(len(df_seg)),
                                      vc_formula={site_col: f"0 + C({site_col})"} if site_col else None, 
                                      data=df_seg)
        try:
            result = mdf.fit()
        except np.linalg.LinAlgError:
            logging.error(f"LinAlgError for segment {s} when fitting LMM, continuing...")
            continue
                        
        for v in result.params.index:
            pval = result.pvalues[v]
            beta = result.params[v]
            t = result.tvalues[v]
            se = result.bse[v]
            sigma_residual = np.sqrt(result.scale) # residual variance
            d_partial = beta / sigma_residual
            df_lmm.append([bin, v, pval, beta, t, se, sigma_residual, d_partial, n_subj])
        residuals.append([bin] + result.resid.values.tolist())
        
    df_lmm = pd.DataFrame(df_lmm, columns=['Bin', 'Variable', 'p', 'beta', 't', 'se', 
                                           'sigma_residual', 'd_partial','n_subjs'])
    df_residual = pd.DataFrame(residuals,columns=['Bin'] + subjs.tolist())

    # convert profile_s{}_r{} to separate s, r columns
    def rename_bin(df):
        s_r = df['Bin'].str.extract(r'bin_s(\d+)_r(\d+)').astype(int)
        df = df.assign(s=s_r[0], r=s_r[1]).drop(columns=['Bin'])
        return df
    
    return rename_bin(df_lmm), rename_bin(df_residual)

def load_stat_as_grid(df, var_name, stat_name, ns, nr):
    '''Helper function for loading statistics from df_lmm as (ns, nr) grid'''
    df_subset = df.loc[(df['Variable']==var_name)]
    arr = np.full((ns, nr), np.nan).astype(np.float32)
    for _, row in df_subset.iterrows():
        si, ri = row['s'], row['r']
        if si < ns and ri < nr:
            arr[si, ri] = row[stat_name]
    return arr

def load_resid_as_grid(df, ns, nr):
    '''Helper function for loading residuals from df_residual as (ns, nr, n_subj) grid'''
    subjs = [c for c in df.columns if c not in ['s','r']]
    arr = np.full((ns, nr, len(subjs)), np.nan).astype(np.float32)
    for i, row in df.iterrows():
        si, ri = int(row['s']), int(row['r'])
        if si < ns and ri < nr:
            arr[si, ri, :] = row[subjs].values
    return arr