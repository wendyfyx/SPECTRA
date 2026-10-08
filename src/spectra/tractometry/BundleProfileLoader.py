"""
BundleDataProvider: Data loading with caching, effect injection, and label shuffling
for tractometry analysis.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from collections import Counter
from functools import cached_property

import numpy as np
import pandas as pd
from tqdm import tqdm

from spectra.simulation.synthetic_effects import compute_std_map, create_effect_map_multi, inject_effect_to_array
from spectra.tractometry.results_data import compute_pooled_cohen_d
from spectra.utils.utils import clean_mask, get_rng

def load_profile(fpath: Path, mode: str, file_ext: str) -> Optional[np.ndarray]:
    """
    Load a single profile from file.
    
    Parameters
    ----------
    fpath : Path
        Path to the profile file (without extension)
    mode : str
        '1d' or '2d' for profile mode
    file_ext : str
        File extension: 'npz', 'npy', or 'txt'
    
    Returns
    -------
    profile : np.ndarray or None
        Profile array of shape (ns,) for 1d or (ns, nr) for 2d
    """
    fpath = Path(str(fpath) + f'.{file_ext}')
    if not fpath.is_file():
        return None
    
    if file_ext == 'npz':
        data = np.load(fpath)
        return data[f'profile_{mode}']
    elif file_ext == 'npy':
        return np.load(fpath)
    elif file_ext == 'txt':
        return np.loadtxt(fpath)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")


def aggregate_profiles(
    metric: str, 
    bundle: str, 
    root_dir: str, 
    subj_list: Optional[List[str]], 
    mode: str = '2d', 
    min_prop: float = 0.3,
    file_ext: str = 'npz',
    disable_tqdm: bool = True,
    return_raw: bool = False,
    edge_mask: Optional[Union[int, Tuple[int, int]]] = None,
) -> Union[Tuple[np.ndarray, np.ndarray, pd.DataFrame], 
           Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]]:
    """
    Aggregate tractometry profiles for a given metric, bundle and subject list.
    
    Assumes profiles are organized as: [root_dir]/[metric]/[bundle]/[subject].[file_ext]
    
    Parameters
    ----------
    metric : str
        Metric name, e.g., 'fa'
    bundle : str
        Bundle name, e.g., 'AF_L'
    root_dir : str
        Root directory containing the metric/bundle folders
    subj_list : list or None
        List of subjects to aggregate profiles for. If None, use all available.
    mode : str
        '1d' or '2d' for profile mode
    min_prop : float
        Minimum proportion of subjects to include in mask
    file_ext : str
        File extension: 'npz', 'npy', or 'txt'
    disable_tqdm : bool
        Disable progress bar
    return_raw : bool
        If True, also return the raw 3D data array (n_subj, ns, nr)
    edge_mask : int or tuple of int, optional
        Number of edge grid cells to mask out. For 1D profiles, an int N masks
        the first N and last N segments along s. For 2D profiles, a tuple (N_s, N_r)
        masks N_s cells on each end along s and N_r cells on each end along r.
        If an int is provided for 2D, it applies only to the s dimension (N_r=0).

    Returns
    -------
    group_avg : np.ndarray, shape (ns, nr)
        Group average profile
    mask : np.ndarray, shape (ns, nr)
        Profile mask (grid cells with insufficient subjects are masked)
    df : pd.DataFrame
        Dataframe with aggregated bundle profiles
    raw_data : np.ndarray, shape (n_subj, ns, nr), only if return_raw=True
        Raw profile data for all subjects
    """
    bundle_dir = Path(root_dir) / metric / bundle
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Directory {bundle_dir} does not exist")
    
    # Get subject list from directory if not provided
    if subj_list is None:
        subj_list = [f.stem for f in bundle_dir.glob(f"*.{file_ext}")]
    
    data = []
    subjects = []
    
    for s in tqdm(subj_list, desc=f"Loading {bundle}", disable=disable_tqdm):
        fpath = bundle_dir / s
        profile = load_profile(fpath, mode, file_ext)
        if profile is not None:
            data.append(profile)
            subjects.append(s)

    if not data:
        raise ValueError(f"No profiles found for bundle {bundle}")

    # Stack and handle dimensions
    raw_data = np.stack(data)  # shape: (N, ns) or (N, ns, nr)
    if raw_data.ndim == 2:
        raw_data = np.expand_dims(raw_data, axis=2)  # (N, ns, 1)
    
    # Create mask and group average
    mask = clean_mask((raw_data != 0).mean(axis=0) >= min_prop)
    
    # Apply edge masking if specified
    if edge_mask is not None:
        n_subj, ns, nr = raw_data.shape
        
        # Parse edge_mask parameter
        if isinstance(edge_mask, int):
            edge_s = edge_mask
            edge_r = 0
        else:
            edge_s, edge_r = edge_mask
        
        # Mask edges along s dimension
        if edge_s > 0:
            if edge_s * 2 >= ns:
                logging.warning(f"edge_mask along s ({edge_s}) too large for grid size ({ns}), "
                              f"masking would remove all cells")
            else:
                mask[:edge_s, :] = False
                mask[-edge_s:, :] = False
        
        # Mask edges along r dimension (only for 2D)
        if edge_r > 0 and nr > 1:
            if edge_r * 2 >= nr:
                logging.warning(f"edge_mask along r ({edge_r}) too large for grid size ({nr}), "
                              f"masking would remove all cells")
            else:
                mask[:, :edge_r] = False
                mask[:, -edge_r:] = False
    
    group_avg = raw_data.mean(axis=0)
    group_avg[~mask] = 0

    # Flatten into dataframe
    n_subj, ns, nr = raw_data.shape
    s_idx, r_idx = np.where(mask)
    col_names = [f'bin_s{s}_r{r}' for s, r in zip(s_idx, r_idx)]

    # Build DataFrame all at once to avoid fragmentation
    data_dict = {'subject': subjects}
    data_dict.update({
        f'bin_s{s}_r{r}': raw_data[:, s, r]
        for s, r in zip(s_idx, r_idx)
    })
    df = pd.DataFrame(data_dict)

    # Ensure numeric columns are float
    for col in df.columns[1:]:
        df[col] = df[col].astype(float)

    logging.info(f"Aggregated {mode} {metric} profiles for {bundle}: "
                 f"{len(df)} subjects, grid ({ns}, {nr})")

    if return_raw:
        return group_avg, mask, df, raw_data
    return group_avg, mask, df


class BundleProfileLoader:
    """
    Data provider for tractometry analysis with caching and effect injection.
    
    Handles:
    - Loading profile data from disk (cached after first load)
    - Merging with metadata
    - Subject subsampling (random or stratified)
    - Shuffling target labels for calibration/null testing
    - Injecting synthetic effects for power analysis
    
    Parameters
    ----------
    metadata_df : pd.DataFrame
        Subject-level metadata
    bundles : List[str]
        Bundle names to analyze
    metric : str
        Metric name (e.g., 'fa')
    root_dir : str
        Root directory for profiles
    mode : str
        '1d' or '2d' profile mode
    file_ext : str
        File extension: 'npz', 'npy', or 'txt'
    min_prop : float
        Minimum proportion of subjects for valid bin
    subj_col : str
        Subject ID column in metadata
    target_col : str
        Target variable column
    target_ref_group : str
        Reference group for categorical target
    target_case_group : str
        Case group for categorical target
    group_col : str
        Group column for random effects
    covar_cols : List[str]
        Covariate columns
    cont_cols : List[str]
        Continuous columns to standardize
    require_common : bool
        If True (default), use only subjects with data for ALL bundles.
        If False, each bundle uses all available subjects for that bundle.
    disable_tqdm : bool
        Disable progress bars
    edge_mask : int or tuple of int, optional
        Number of edge grid cells to mask out. 
    """
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        bundles: List[str],
        metric: str,
        root_dir: str,
        mode: str = '2d',
        file_ext: str = 'npz',
        min_prop: float = 0.3,
        subj_col: str = 'subject',
        target_col: str = 'dx',
        target_ref_group: Optional[Any] = None,
        target_case_group: Optional[Any] = None,
        group_col: Optional[str] = 'site',
        covar_cols: Optional[List[str]] = None,
        cont_cols: Optional[List[str]] = None,
        require_common: bool = False,
        disable_tqdm: bool = False,
        edge_mask: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        self.bundles = bundles
        self.metric = metric
        self.root_dir = root_dir
        self.mode = mode
        self.file_ext = file_ext
        self.min_prop = min_prop
        self.subj_col = subj_col
        self.target_col = target_col
        self.target_ref_group = target_ref_group
        self.target_case_group = target_case_group
        self.group_col = group_col
        self.covar_cols = covar_cols or ['age', 'sex']
        self.cont_cols = cont_cols or ['age']
        self.require_common = require_common
        self.disable_tqdm = disable_tqdm
        self.edge_mask = edge_mask
        
        # Store original metadata for reset
        self._original_metadata = metadata_df.copy()
        self.metadata_df = metadata_df.copy()
        
        # Filter metadata: drop rows with missing required columns
        self._filter_metadata()
        
        # Cache for loaded data: {bundle: (group_avg, mask, df, raw_data)}
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]] = {}
        
        # Track state for reset
        self._has_injected_effects = False
        self._injected_columns: List[str] = []
        self._current_subjects: Optional[np.ndarray] = None  # None means use all common_subjects
    
    def _filter_metadata(self):
        """Filter metadata to rows with valid values for required columns."""
        required_cols = [self.target_col, self.subj_col] + self.covar_cols
        if self.group_col:
            required_cols.append(self.group_col)
        
        n_before = len(self.metadata_df)
        self.metadata_df = self.metadata_df.dropna(subset=required_cols)
        
        # Filter to target groups if categorical
        if self.target_ref_group is not None:
            valid_groups = [self.target_ref_group]
            if self.target_case_group is not None:
                valid_groups.append(self.target_case_group)
            self.metadata_df = self.metadata_df[
                self.metadata_df[self.target_col].isin(valid_groups)
            ]
        logging.info(f"Filtered metadata: {n_before} -> {len(self.metadata_df)} rows "
                    f"(required cols: {required_cols})")
    
    @cached_property
    def subjects_by_bundle(self) -> Dict[str, np.ndarray]:
        """
        Get available subjects for each bundle (cached).
        
        Returns dict mapping bundle name to array of subject IDs that have
        data for that bundle AND are in metadata.
        """
        meta_subjs = set(self.metadata_df[self.subj_col].values)
        result = {}
        
        for bundle in self.bundles:
            bundle_dir = Path(self.root_dir) / self.metric / bundle
            if not bundle_dir.exists():
                raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
            
            bundle_subjs = set(f.stem for f in bundle_dir.glob(f"*.{self.file_ext}"))
            # Intersect with metadata
            available = bundle_subjs & meta_subjs
            result[bundle] = np.array(sorted(list(available)))
            
        logging.info(f"Subjects per bundle: " + 
                    ", ".join(f"{b}={len(s)}" for b, s in result.items()))
        return result

    @cached_property
    def all_subjects(self) -> np.ndarray:
        """
        Union of subjects across all bundles (cached).
        
        Returns subjects that have data for AT LEAST ONE bundle AND are in metadata.
        """
        union = set()
        for bundle_subjs in self.subjects_by_bundle.values():
            union.update(bundle_subjs)
        
        logging.info(f"Found {len(union)} total subjects (union across {len(self.bundles)} bundles)")
        return np.array(sorted(list(union)))

    @cached_property
    def common_subjects(self) -> np.ndarray:
        """Find subjects with data for ALL bundles (cached)."""
        common = None
        for bundle in self.bundles:
            bundle_dir = Path(self.root_dir) / self.metric / bundle
            if not bundle_dir.exists():
                raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
            bundle_subjs = set(f.stem for f in bundle_dir.glob(f"*.{self.file_ext}"))
            
            if common is None:
                common = bundle_subjs
            else:
                common = common & bundle_subjs
        
        # Intersect with metadata subjects
        meta_subjs = set(self.metadata_df[self.subj_col].values)
        common = common & meta_subjs
        
        logging.info(f"Found {len(common)} common subjects across {len(self.bundles)} bundles")
        return np.array(sorted(list(common)))
    
    @property
    def subjects(self) -> np.ndarray:
        """Get current active subjects (after any subsampling)."""
        if self._current_subjects is not None:
            return self._current_subjects
        if self.require_common:
            return self.common_subjects
        else:
            return self.all_subjects
    
    @property
    def n_subjects(self) -> int:
        """Number of currently active subjects."""
        return len(self.subjects)

    def get_subjects_for_bundle(self, bundle: str) -> np.ndarray:
        """
        Get subjects available for a specific bundle, respecting current subsample.
        
        Parameters
        ----------
        bundle : str
            Bundle name
            
        Returns
        -------
        np.ndarray
            Subject IDs that have data for this bundle AND are in current active set.
        """
        bundle_subjs = self.subjects_by_bundle[bundle]
        
        if self._current_subjects is not None:
            return np.intersect1d(self._current_subjects, bundle_subjs)
        elif self.require_common:
            return self.common_subjects
        else:
            return bundle_subjs
    
    def get_groups_for_bundle(self, bundle: str) -> np.ndarray:
        """
        Get group labels for subjects in a specific bundle.
        
        Parameters
        ----------
        bundle : str
            Bundle name
            
        Returns
        -------
        np.ndarray
            Group labels in same order as get_subjects_for_bundle(bundle)
        """
        bundle_subjs = self.get_subjects_for_bundle(bundle)
        meta = self.metadata_df[self.metadata_df[self.subj_col].isin(bundle_subjs)]
        # Ensure order matches bundle_subjs
        meta = meta.set_index(self.subj_col).loc[bundle_subjs]
        
        if self.group_col and self.group_col in meta.columns:
            return meta[self.group_col].values
        else:
            return np.zeros(len(bundle_subjs), dtype=int)
    
    def get_n_subjects_by_bundle(self) -> Dict[str, int]:
        """Get number of subjects per bundle (respecting current subsample)."""
        return {bundle: len(self.get_subjects_for_bundle(bundle)) 
                for bundle in self.bundles}

    def subsample(
        self,
        n: int,
        stratify_by: Optional[str] = None,
        min_per_group: int = 10,
        replace: bool = False,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> np.ndarray:
        """
        Subsample subjects from the subject pool.
        
        If require_common=True, samples from common_subjects.
        If require_common=False, samples from all_subjects.
        
        Parameters
        ----------
        n : int
            Number of subjects to sample
        stratify_by : str, optional
            Column to stratify by (e.g., target_col to maintain case/control ratio).
            If None, samples randomly without stratification.
        min_per_group : int
            When stratify_by is set, minimum subjects to sample from each group.
        replace : bool
            Whether to sample with replacement
        rng : np.random.Generator or int, optional
            Random generator or seed
            
        Returns
        -------
        sampled_subjects : np.ndarray
            Array of sampled subject IDs
        """
        rng = get_rng(rng)
        
        # Determine pool based on require_common
        if self.require_common:
            pool = self.common_subjects
        else:
            pool = self.all_subjects
        
        # Get subjects with metadata
        meta_subjs = self.metadata_df[self.metadata_df[self.subj_col].isin(pool)]
        available_n = len(meta_subjs)
        
        if n > available_n and not replace:
            logging.warning(f"Requested {n} subjects but only {available_n} available. "
                          f"Using all {available_n}.")
            n = available_n
        
        if stratify_by is None:
            # Simple random sampling
            idx = rng.choice(available_n, size=n, replace=replace)
            sampled = meta_subjs[self.subj_col].values[idx]
        else:
            # Stratified sampling
            sampled = self._stratified_sample(
                meta_subjs, n, stratify_by, replace, rng, min_per_group
            )
            counter = Counter(meta_subjs.loc[meta_subjs[self.subj_col].isin(sampled)][stratify_by].values)
        
        self._current_subjects = np.array(sampled)
        logging.info(f"Subsampled {len(self._current_subjects)} subjects" + 
                    (f" (stratified by {stratify_by}, {counter})" if stratify_by else ""))
        
        return self._current_subjects
    
    def _stratified_sample(
        self,
        df: pd.DataFrame,
        n: int,
        stratify_by: str,
        replace: bool,
        rng: np.random.Generator,
        min_per_group: int = 5,
    ) -> List[str]:
        """
        Stratified sampling with optional minimum per group.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to sample from
        n : int
            Total number to sample
        stratify_by : str
            Column to stratify by
        replace : bool
            Sample with replacement
        rng : np.random.Generator
            Random generator
        min_per_group : int
            Minimum subjects per group. Groups with fewer available subjects
            will contribute all they have (if min_per_group > available).
            Remaining slots are distributed proportionally.
        
        Returns
        -------
        List[str]
            Sampled subject IDs
        """
        groups = list(df[stratify_by].unique())
        group_counts = df[stratify_by].value_counts()
        n_groups = len(groups)
        
        # Calculate allocation per group
        allocation = {}
        
        if min_per_group > 0:
            # First pass: allocate minimum to each group (capped by availability)
            reserved = 0
            for group in groups:
                n_available = group_counts[group]
                alloc = min(min_per_group, n_available) if not replace else min_per_group
                allocation[group] = alloc
                reserved += alloc
            
            # Check if minimum allocation exceeds total requested
            if reserved > n:
                logging.warning(f"min_per_group={min_per_group} with {n_groups} groups "
                              f"requires {reserved} subjects, but only {n} requested. "
                              f"Reducing proportionally.")
                # Scale down proportionally
                scale = n / reserved
                for group in groups:
                    allocation[group] = max(1, int(allocation[group] * scale))
                # Adjust to hit exactly n
                while sum(allocation.values()) > n:
                    # Remove from largest group
                    largest = max(allocation, key=allocation.get)
                    if allocation[largest] > 1:
                        allocation[largest] -= 1
            else:
                # Distribute remaining slots proportionally among groups
                remaining = n - reserved
                if remaining > 0:
                    # Proportional weights based on remaining capacity
                    weights = {}
                    total_weight = 0
                    for group in groups:
                        n_available = group_counts[group]
                        already_allocated = allocation[group]
                        capacity = (n_available - already_allocated) if not replace else float('inf')
                        weights[group] = max(0, min(capacity, group_counts[group]))
                        total_weight += weights[group]
                    
                    if total_weight > 0:
                        for i, group in enumerate(groups):
                            if i == n_groups - 1:
                                # Last group gets remainder
                                extra = remaining
                            else:
                                extra = int(np.round(remaining * weights[group] / total_weight))
                                extra = min(extra, remaining)
                            
                            # Cap by available capacity
                            if not replace:
                                n_available = group_counts[group]
                                max_extra = n_available - allocation[group]
                                extra = min(extra, max_extra)
                            
                            allocation[group] += extra
                            remaining -= extra
        else:
            # Original proportional allocation
            total = group_counts.sum()
            remaining = n
            for i, group in enumerate(groups):
                if i == n_groups - 1:
                    allocation[group] = remaining
                else:
                    alloc = int(np.round(n * group_counts[group] / total))
                    alloc = min(alloc, remaining)
                    allocation[group] = alloc
                    remaining -= alloc
        
        # Sample from each group
        sampled = []
        for group in groups:
            group_df = df[df[stratify_by] == group]
            n_group = allocation[group]
            n_available = len(group_df)
            
            if n_group > n_available and not replace:
                n_group = n_available
            
            if n_group > 0:
                idx = rng.choice(n_available, size=n_group, replace=replace)
                sampled.extend(group_df[self.subj_col].values[idx])
        
        return sampled
    
    def subsample_balanced(
        self,
        n_per_group: int,
        group_col: Optional[str] = None,
        replace: bool = False,
        rng: Optional[Union[np.random.Generator, int]] = None,
    ) -> np.ndarray:
        """
        Subsample equal numbers from each group (e.g., balanced case/control).
        
        Parameters
        ----------
        n_per_group : int
            Number of subjects to sample from each group
        group_col : str, optional
            Column defining groups. Defaults to target_col (note it's different from
            self.group_col which refers to the random effect columm)
        replace : bool
            Whether to sample with replacement
        rng : np.random.Generator or int, optional
            Random generator or seed
            
        Returns
        -------
        sampled_subjects : np.ndarray
        """
        rng = get_rng(rng)
        
        if group_col is None:
            group_col = self.target_col

        if self.require_common:
            pool = self.common_subjects
        else:
            pool = self.all_subjects
        
        meta_subjs = self.metadata_df[self.metadata_df[self.subj_col].isin(pool)]
        groups = meta_subjs[group_col].unique()
        
        sampled = []
        for group in groups:
            group_df = meta_subjs[meta_subjs[group_col] == group]
            n_available = len(group_df)
            
            n_sample = n_per_group
            if n_sample > n_available and not replace:
                logging.warning(f"Group '{group}' has only {n_available} subjects, "
                              f"sampling all instead of {n_per_group}")
                n_sample = n_available
            
            idx = rng.choice(n_available, size=n_sample, replace=replace)
            sampled.extend(group_df[self.subj_col].values[idx])
        
        self._current_subjects = np.array(sampled)
        logging.info(f"Balanced subsample: {n_per_group} per group, "
                    f"{len(self._current_subjects)} total")
        
        return self._current_subjects
    
    def set_subjects(self, subjects: Union[List[str], np.ndarray]):
        """
        Manually set the active subjects.
        
        Parameters
        ----------
        subjects : array-like
            Subject IDs to use
        """
        subjects = np.array(subjects)
        
        # Validate subjects exist in appropriate pool
        if self.require_common:
            valid_pool = self.common_subjects
        else:
            valid_pool = self.all_subjects

        valid = np.isin(subjects, valid_pool)
        if not valid.all():
            invalid = subjects[~valid]
            raise ValueError(f"Found ({len(invalid)}) invalid subjects: {invalid[:5]}...")
        
        self._current_subjects = subjects
        logging.info(f"Set {len(subjects)} active subjects")
    
    def reset_subjects(self):
        """Reset to using all subjects."""
        self._current_subjects = None
    
    def get_bundle_data(
        self, 
        bundle: str,
        subj_list: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Get bundle data, loading from cache if available.
        
        Parameters
        ----------
        bundle : str
            Bundle name
        subj_list : np.ndarray, optional
            Subset of subjects. If None, uses current active subjects (after subsampling).
        
        Returns
        -------
        group_avg : np.ndarray (ns, nr)
        mask : np.ndarray (ns, nr)
        df : pd.DataFrame with profile data merged with metadata
        """
        if subj_list is None:
            subj_list = self.get_subjects_for_bundle(bundle)
        
        # Load and cache if not already cached
        if bundle not in self._cache:
            self._load_bundle(bundle)
        group_avg, mask, df_profile, raw_data = self._cache[bundle]
        
        # Filter to requested subjects
        df_filtered = df_profile[df_profile['subject'].isin(subj_list)].copy()
        
        # Merge with metadata
        df_merged = df_filtered.merge(
            self.metadata_df, 
            left_on='subject', 
            right_on=self.subj_col, 
            how='inner'
        )
        if 'subject' in df_merged.columns and self.subj_col != 'subject':
            df_merged = df_merged.drop(columns='subject')
        
        # Set categorical target reference level
        if self.target_ref_group is not None and self.target_col in df_merged.columns:
            col_dtype = df_merged[self.target_col].dtype
            if df_merged[self.target_col].dtype == 'object' or pd.api.types.is_integer_dtype(col_dtype):
                categories = [self.target_ref_group] + [
                    g for g in df_merged[self.target_col].unique() 
                    if g != self.target_ref_group
                ]
                df_merged[self.target_col] = pd.Categorical(
                    df_merged[self.target_col], categories=categories
                )

        # Convert numpy.str_ to Python native str (for pymer4)
        for col in df_merged.select_dtypes(include=['object']).columns:
            df_merged[col] = df_merged[col].apply(str)
        
        return group_avg, mask, df_merged
    
    def _load_bundle(self, bundle: str):
        """Load bundle data from disk and cache it."""
        group_avg, mask, df, raw_data = aggregate_profiles(
            metric=self.metric,
            bundle=bundle,
            root_dir=self.root_dir,
            subj_list=list(self.subjects_by_bundle[bundle]),
            mode=self.mode,
            min_prop=self.min_prop,
            file_ext=self.file_ext,
            disable_tqdm=self.disable_tqdm,
            return_raw=True,
            edge_mask=self.edge_mask,
        )
        
        n_subj, ns, nr = raw_data.shape
        logging.debug(f"Cached {bundle}: {len(df)} subjects, grid ({ns}, {nr})")
        self._cache[bundle] = (group_avg, mask, df, raw_data)
    
    def get_raw_data(self, bundle: str) -> np.ndarray:
        """Get raw 3D profile data (n_subj, ns, nr) for a bundle."""
        if bundle not in self._cache:
            self._load_bundle(bundle)
        return self._cache[bundle][3].copy()
    
    def get_residual_std(self, bundle: str) -> np.ndarray:
        """
        Estimate residual std at each grid cell for effect size scaling.
        
        Returns pooled std across subjects at each (s, r) location.
        """
        return compute_std_map(self.get_raw_data(bundle))

    
    def inject_synthetic_effect(
        self,
        effect_maps: Dict[str, np.ndarray],
        effect_size: float,
        effect_subjects: Union[List[str], np.ndarray],
        effect_col_name: str = 'synthetic_effect',
        scale_by_residual: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Inject synthetic effects into cached data for specified subjects.
        
        For subjects in effect_subjects, adds:
            data[subj, s, r] += effect_map[s, r] * effect_size * std[s, r]
        
        Subjects not found in a bundle (e.g., missing data) are silently skipped.
        
        Parameters
        ----------
        effect_maps : Dict[str, np.ndarray]
            {bundle: effect_map} with values typically in [0, 1] indicating 
            spatial pattern of effect magnitude
        effect_size : float
            Cohen's d to inject (scaled by local std if scale_by_residual=True)
        effect_subjects : array-like
            Subject IDs that should receive the effect (e.g., "case" subjects).
            Subjects missing from a bundle are skipped for that bundle.
        effect_col_name : str
            Column name for binary effect indicator added to metadata
        scale_by_residual : bool
            If True, scale effect by local residual std for interpretable Cohen's d
        
        Returns
        -------
        true_d_maps : Dict[str, np.ndarray]
            {bundle: true_cohen_d} where true_cohen_d is the realized Cohen's d
            at each grid cell, computed from post-injection data using the same
            pooled within-group SD denominator as MultiBundleLM._compute_cohen_d.
            Shape (ns, nr) per bundle; NaN where mask is False.
        """
        effect_subjects = set(np.asarray(effect_subjects))
        
        # Add binary effect column to metadata
        self.metadata_df[effect_col_name] = (
            self.metadata_df[self.subj_col].isin(effect_subjects).astype(int)
        )
        self._injected_columns.append(effect_col_name)
        
        # Track total affected subjects across bundles
        all_affected = set()
        true_d_maps = {}
        
        for bundle in self.bundles:
            if bundle not in effect_maps:
                logging.warning(f"No effect map for {bundle}, skipping")
                continue
            
            # Load bundle data if not cached
            if bundle not in self._cache:
                self._load_bundle(bundle)
            group_avg, mask, df, raw_data = self._cache[bundle]
            cached_subjects = df['subject'].values
            
            # Create boolean mask: which cached subjects get the effect
            subject_mask = np.array([s in effect_subjects for s in cached_subjects])
            if not subject_mask.any():
                continue
            all_affected.update(cached_subjects[subject_mask])
            
            # Get and validate effect map shape
            effect_map = effect_maps[bundle]
            if effect_map.ndim == 1 and raw_data.ndim == 3 and raw_data.shape[2] == 1:
                effect_map = effect_map[:, np.newaxis]  # (ns,) -> (ns, 1)
            if effect_map.shape != raw_data.shape[1:]:
                raise ValueError(f"Effect map shape {effect_map.shape} doesn't match "
                               f"data shape {raw_data.shape[1:]} for {bundle}")
            
            # Inject effect
            std_map = compute_std_map(raw_data) if scale_by_residual else None
            raw_data = inject_effect_to_array(
                raw_data, effect_map, effect_size, subject_mask, std_map
            )
            
            # Compute true Cohen's d from post-injection data (zeros = missing)
            ns, nr = mask.shape if mask.ndim == 2 else (mask.shape[0], 1)
            true_d = np.full((ns, nr), np.nan)
            observed_data = np.where(raw_data != 0, raw_data, np.nan)
            case_data = observed_data[subject_mask]
            control_data = observed_data[~subject_mask]
            
            # Compute Cohen's d at each grid cell using shared utility
            for s in range(ns):
                for r in range(nr):
                    if mask[s, r]:
                        true_d[s, r] = compute_pooled_cohen_d(
                            case_data[:, s, r],
                            control_data[:, s, r]
                        )
            true_d_maps[bundle] = true_d
            
            # Rebuild DataFrame from modified data
            s_idx, r_idx = np.where(mask)
            data_dict = {'subject': cached_subjects}
            data_dict.update({
                f'bin_s{s}_r{r}': raw_data[:, s, r]
                for s, r in zip(s_idx, r_idx)
            })
            df_new = pd.DataFrame(data_dict)
            
            # Update cache
            group_avg_new = raw_data.mean(axis=0)
            group_avg_new[~mask] = 0
            self._cache[bundle] = (group_avg_new, mask, df_new, raw_data)
        
        self._has_injected_effects = True
        logging.info(f"Injected effect (d={effect_size}) to {len(all_affected)} subjects "
                    f"across {len(self.bundles)} bundles")
        return true_d_maps
    
    def shuffle_column(
        self, 
        col_name: str, 
        rng: Optional[Union[np.random.Generator, int]] = None
    ):
        """
        Shuffle a column in metadata (e.g., target labels for null testing).
        
        Parameters
        ----------
        col_name : str
            Column to shuffle
        rng : np.random.Generator or int, optional
            Random generator or seed
        """
        rng = get_rng(rng)
        self.metadata_df[col_name] = rng.permutation(self.metadata_df[col_name].values)
        logging.debug(f"Shuffled column '{col_name}'")
    
    def reset(self):
        """
        Reset to original state: restore metadata, clear effects, reset subjects.
        """
        self.metadata_df = self._original_metadata.copy()
        self._filter_metadata()  # Re-apply filtering
        
        if self._has_injected_effects:
            # Clear cache to force reload from disk
            self._cache.clear()
            self._has_injected_effects = False
        
        self._injected_columns.clear()
        self._current_subjects = None
        logging.debug("Reset provider to original state")
    
    def clear_cache(self):
        """Clear all cached data (forces reload from disk)."""
        self._cache.clear()
        logging.debug("Cleared data cache")
    
    def get_metadata(self) -> pd.DataFrame:
        """Get current metadata dataframe."""
        return self.metadata_df.copy()
    
    def get_grid_info(self, bundle: str) -> Tuple[int, int, np.ndarray]:
        """
        Get grid dimensions and mask for a bundle.
        
        Parameters
        ----------
        bundle : str
            Bundle name
            
        Returns
        -------
        ns, nr : int
            Grid dimensions
        mask : np.ndarray (ns, nr)
            Valid grid cells
        """
        if bundle not in self._cache:
            self._load_bundle(bundle)
        _, mask, _, raw_data = self._cache[bundle]
        ns, nr = raw_data.shape[1:]
        return ns, nr, mask
    
    def create_effect_maps(
        self,
        n_blobs: int = 1,
        s_len: float = 3.0,
        r_len: float = 3.0,
        rng: Optional[Union[np.random.Generator, int]] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Create effect maps for all bundles.
        
        For a single effect region, pass n_blobs=1 (default) with kwargs
        for create_effect_map (center, extent, max_effect, min_effect,
        decay, geometry). For multiple random regions, set n_blobs > 1
        with center_range, extent_range, max_effect_range.
        
        Parameters
        ----------
        n_blobs : int
            Number of effect regions. If 1, calls create_effect_map
            directly. If > 1, calls create_effect_map_multi.
        s_len, r_len : float
            Grid spacing in mm.
        rng : np.random.Generator or int, optional
            Random generator (for multi-blob generation).
        **kwargs
            Passed to create_effect_map or create_effect_map_multi
            (center_range, extent_range, max_effect_range, 
            min_effect, decay, geometry)

        Returns
        -------
        effect_maps : Dict[str, np.ndarray]
            {bundle: effect_map} for each bundle
        """
        effect_maps = {}
        
        for bundle in self.bundles:
            ns, nr, mask = self.get_grid_info(bundle)
            
            effect_maps[bundle] = create_effect_map_multi(
                ns, nr,
                n_blobs=n_blobs,
                s_len=s_len,
                r_len=r_len,
                mask=mask,
                rng=rng,
                **kwargs,
            )
        
        return effect_maps