import logging

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA

from nibabel.streamlines.array_sequence import ArraySequence
from dipy.tracking.utils import length
from dipy.tracking.streamline import set_number_of_points, Streamlines, \
                                     values_from_volume, orient_by_streamline
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundles

MERGE_THRESHOLD=0.25
MASK_THRESHOLD=0.25

def get_centroid(bundle, n_points=50, thresh=100):
    '''Get bundle centroid(s) from a bundle with a given distance threshold'''
    if isinstance(bundle, np.ndarray):
        bundle = ArraySequence(bundle[:,:,:3])

    ref_bundle = set_number_of_points(bundle, nb_points=n_points)
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=thresh, metric=metric)
    clusters = qb.cluster(ref_bundle)
    centroids = Streamlines(clusters.centroids)
    lens = np.asarray(list(length(centroids)))
    return centroids[np.argmax(lens)]

def get_n_segment_by_length(atlas_bundle, s_len=5):
    '''Calculate the number of segments given segment length (mm)'''
    avg_length = np.mean(list(length(atlas_bundle)))
    return max(int(np.round(avg_length / s_len)), 1)

def compute_robust_centroid(
    bundle,
    s_len: float = 5.0,
    thresh: float = 100.0,
    extrapolate_prop: float = 0.3,
    method: str = 'linear',
) -> np.ndarray:
    """
    Compute a robust centroid that extends to the full bundle extent.
    This addresses the problem where the QuickBundles centroid is shorter
    than the longer streamlines, causing very large end segments.

    Parameters
    ----------
    bundle : ArraySequence
        Atlas bundle
    s_len : float
        Target along-tract segment size (mm). Used to compute ns from
        the extended arc length.
    thresh : float
        QuickBundles distance threshold
    extrapolate_prop : float
        The proportion of the QB centroid arc length to extrapolate by
        on each end. Default: 0.3.
    method : str
        Extrapolation method. Either 'spline' or 'linear' (default).
        'linear' extrapolates from the terminal tangent vectors of the QB
        centroid, more robust when there's branching at the end
        'spline' fits a cubic spline through the QB centroid and evaluates
        it beyond [0, 1]

    Returns
    -------
    robust_centroid : np.ndarray, shape (ns, 3)
        Extended centroid coordinates, evenly spaced in arc-length,
        compatible with get_centroid output.
    """
    if method not in ('spline', 'linear'):
        raise ValueError(f"method must be 'spline' or 'linear', got '{method}'")

    # initialize with QuickBundles centroid
    initial_centroid = get_centroid(bundle, n_points=100, thresh=thresh)
    initial_centroid = np.array(initial_centroid)

    # fallback: QB centroid resampled to the correct ns
    def _fallback():
        ns_fallback = get_n_segment_by_length(bundle, s_len=s_len)
        return np.array(get_centroid(bundle, n_points=ns_fallback, thresh=thresh))

    # approximate arc length from QB centroid points
    centroid_arc_length = float(
        np.linalg.norm(np.diff(initial_centroid, axis=0), axis=1).sum()
    )
    ext_mm = extrapolate_prop * centroid_arc_length

    if method == 'spline':
        if len(initial_centroid) < 4:
            logging.warning(
                "compute_robust_centroid: initial centroid has fewer than 4 points, "
                "cannot fit cubic spline. Falling back."
            )
            return _fallback()

        # fit parametric spline through centroid
        tck, u = splprep(initial_centroid.T, s=0, k=3)

        # convert ext_mm to t-units via terminal derivatives (ds/dt at boundaries)
        deriv_start = np.linalg.norm(np.array(splev(0.0, tck, der=1)))
        deriv_end   = np.linalg.norm(np.array(splev(1.0, tck, der=1)))
        t_ext_start = ext_mm / deriv_start if deriv_start > 0 else 0.1
        t_ext_end   = ext_mm / deriv_end   if deriv_end   > 0 else 0.1

        # single dense sampling over extended t range
        # splev extrapolates linearly outside [0, 1] following terminal curvature
        t_dense = np.linspace(0.0 - t_ext_start, 1.0 + t_ext_end, 1000)
        ext_pts = np.array(splev(t_dense, tck)).T   # (1000, 3)

    else:  # linear
        # extrapolate from terminal tangent vectors of the QB centroid
        tangent_start = initial_centroid[0] - initial_centroid[1]
        tangent_end   = initial_centroid[-1] - initial_centroid[-2]
        tangent_start /= np.linalg.norm(tangent_start)
        tangent_end   /= np.linalg.norm(tangent_end)

        # build n_ext extension points on each side, spaced ~s_len apart
        n_ext = max(1, int(np.round(ext_mm / s_len)))
        steps = np.arange(1, n_ext + 1)[:, np.newaxis]  # (n_ext, 1)
        ext_start_pts = initial_centroid[0] + steps * tangent_start * s_len  # (n_ext, 3)
        ext_end_pts   = initial_centroid[-1] + steps * tangent_end   * s_len  # (n_ext, 3)

        # prepend/append to centroid (start points in reverse so curve is continuous)
        ext_pts = np.vstack([ext_start_pts[::-1], initial_centroid, ext_end_pts])
        t_dense = None   # not used in linear path

    # build cKDTree on extended curve for endpoint projection
    ext_tree = cKDTree(ext_pts)
    # use integer index directly (works for both spline and linear)
    idx_vals = np.arange(len(ext_pts))

    logging.info(
        f"compute_robust_centroid ({method}): centroid_arc_length={centroid_arc_length:.1f}mm, "
        f"ext_mm={ext_mm:.1f}mm"
    )

    # project streamline endpoints onto the extended curve
    i_starts, i_ends = [], []
    for sl in bundle:
        sl = np.array(sl)
        if len(sl) < 2:
            continue
        _, i_s = ext_tree.query(sl[0])
        _, i_e = ext_tree.query(sl[-1])
        i_starts.append(i_s)
        i_ends.append(i_e)

    if not i_starts:
        logging.warning(
            "compute_robust_centroid: no valid streamline endpoints found. Falling back."
        )
        return _fallback()

    i_starts = np.array(i_starts)
    i_ends   = np.array(i_ends)

    # handle reversed streamlines
    i_min_per_sl = np.minimum(i_starts, i_ends)
    i_max_per_sl = np.maximum(i_starts, i_ends)

    # robust extent via percentiles
    i_start = int(np.round(np.percentile(i_min_per_sl, 2.0)))
    i_end   = int(np.round(np.percentile(i_max_per_sl, 98.0)))

    if i_end <= i_start:
        logging.warning(
            f"compute_robust_centroid: i_start ({i_start}) >= i_end ({i_end}). Falling back."
        )
        return _fallback()

    # compute arc length between i_start and i_end
    ext_pts_range = ext_pts[i_start:i_end + 1]

    if len(ext_pts_range) < 2:
        logging.warning(
            "compute_robust_centroid: too few samples in range. Falling back."
        )
        return _fallback()

    segment_lengths = np.linalg.norm(np.diff(ext_pts_range, axis=0), axis=1)
    arc_length = float(segment_lengths.sum())

    # resample evenly at ns points derived from arc length
    ns = max(int(np.round(arc_length / s_len)), 1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    target_lengths = np.linspace(0.0, arc_length, ns)

    if method == 'spline':
        # interpolate back to t values then evaluate spline for smooth output
        t_range = t_dense[i_start:i_end + 1]
        t_resampled = np.interp(target_lengths, cum_lengths, t_range)
        robust_centroid = np.array(splev(t_resampled, tck)).T
    else:
        # interpolate directly on the extended point array
        robust_centroid = np.array([
            np.interp(target_lengths, cum_lengths, ext_pts_range[:, dim])
            for dim in range(3)
        ]).T

    logging.info(
        f"compute_robust_centroid: extended centroid from {len(initial_centroid)} "
        f"to {ns} points (arc_length={arc_length:.1f}mm, s_len={s_len}mm)"
    )
    return robust_centroid

def create_radial_bins(r_signed, r_len=5):
    """
    Bin signed radial distances based on bin length.
    Merge bins on the edge if they have too few points.

    Parameters
    ----------
    r_signed : np.ndarray, shape (M,)
        Signed radial distances for each point
    r_len : float
        Target radial bin size (mm)

    Returns
    -------
    r_index : np.ndarray, shape (M,)
        Bin index for each point
    nr : int
        Total number of bins after merging
    r_edges: np.nd.array, shape (nr+1, )
        Edges of the radial bins
    """
    r_min, r_max = r_signed.min(), r_signed.max()
    nr = int(np.ceil((r_max - r_min) / r_len))
    r_edges = np.linspace(r_min, r_max, nr + 1)

    # Create r_index
    r_index = np.digitize(r_signed, r_edges) - 1
    r_index = np.clip(r_index, 0, nr - 1)

    # Define point threshold as median point count across bins * 0.2
    point_count = np.bincount(r_index, minlength=nr)
    thresh = np.median(point_count) * MERGE_THRESHOLD

    # Merge edge bins
    merged = False
    if nr > 1 and point_count[0] < thresh:
        # Merge first bin into the second
        r_index[r_index == 0] = 1
        r_edges = r_edges[1:]  # drop first edge
        merged = True

    if nr > 1 and point_count[-1] < thresh:
        # Merge last bin into the previous one
        r_index[r_index == nr - 1] = nr - 2
        r_edges = r_edges[:-1]  # drop last edge
        merged = True

    # Reindex the bins so they start from 0
    _, r_index = np.unique(r_index, return_inverse=True)
    nr_new = r_index.max() + 1

    if merged:
        logging.debug(f"Merged edge bins to create {nr_new} radial bins (original: {nr})")
    return r_index, nr_new, r_edges
    
def get_grid_from_atlas(
    atlas_bundle,
    s_len: float = 5.0,
    r_len: float = 5.0,
    use_robust_centroid: bool = False,
    robust_method: str = 'linear',
):
    """
    Define shared (s, r) grid based on atlas bundle geometry
    along tract and radially across tract.

    Parameters
    ----------
    atlas_bundle : ArraySequence
        Atlas bundle
    s_len : float
        Target along-tract segment size (mm)
    r_len : float
        Target radial bin size (mm)
    use_robust_centroid : bool
        If True, use compute_robust_centroid to extend the centroid to the
        full arc-length extent of the bundle, avoiding large end segments.
        If False (default), uses the standard QuickBundles centroid (BUAN).
    robust_method : str
        Extrapolation method for robust centroid: 'spline' or 'linear' (default).
        Only used when use_robust_centroid=True.

    Returns
    -------
    s_index :  np.ndarray, shape (n_points, )
        Point-wise assignment of along-tract segments
    r_index :  np.ndarray, shape (n_points, )
        Point-wise assignment of radial bins
    centroid :  np.ndarray, shape (ns, 3)
        Atlas centroid coordinates
    uvecs :  np.ndarray, shape (ns, 3)
        Radial vectors for each along-tract segment
    r_edges:  np.ndarray, shape (nr+1, )
        Edges of radial bins
    """

    # Create along-tract segments
    if use_robust_centroid:
        centroid = compute_robust_centroid(
            atlas_bundle,
            s_len=s_len,
            thresh=100,
            extrapolate_prop=0.3,
            method=robust_method,
        )
        ns = len(centroid)
    else:
        ns = get_n_segment_by_length(atlas_bundle, s_len=s_len)
        centroid = get_centroid(atlas_bundle, n_points=ns, thresh=100)

    points = atlas_bundle.get_data()
    _, s_index = cKDTree(centroid).query(points, k=1)

    # Merge terminal segments if they have too few points (same logic as create_radial_bins)
    point_count = np.bincount(s_index, minlength=ns)
    seg_thresh = np.median(point_count) * MERGE_THRESHOLD
    merged = False
    if ns > 1 and point_count[0] < seg_thresh:
        s_index[s_index == 0] = 1
        merged = True
    if ns > 1 and point_count[-1] < seg_thresh:
        s_index[s_index == ns - 1] = ns - 2
        merged = True
    if merged:
        _, s_index = np.unique(s_index, return_inverse=True)
        ns = s_index.max() + 1
        centroid = centroid[:ns]
        logging.debug(f"Merged terminal segments: new ns={ns}")

    # Create radial vectors per segment
    tangents = np.gradient(centroid, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

    u_vecs = np.zeros_like(tangents)
    for s in range(ns):
        seg_pts = points[s_index == s]
        if len(seg_pts) < 3:
            u_vecs[s] = np.array([0,0,0])
            continue
        d = seg_pts - centroid[s]
        # project onto plane orthogonal to tangent
        d_proj = d - np.outer(np.dot(d, tangents[s]), tangents[s])
        pca = PCA(n_components=1).fit(d_proj)
        u_vecs[s] = pca.components_[0]

    # Flip to ensure continuity along-tract
    for s in range(1, ns):
        if np.dot(u_vecs[s], u_vecs[s-1]) < 0:
            u_vecs[s] = -u_vecs[s]

    # Smoothing
    u_vecs = gaussian_filter1d(u_vecs, sigma=1, axis=0)
    u_vecs /= np.linalg.norm(u_vecs, axis=1, keepdims=True)

    # Compute radial bins
    delta = atlas_bundle.get_data() - centroid[s_index]
    r_signed = np.sum(delta * u_vecs[s_index], axis=1)
    r_index, nr, r_edges = create_radial_bins(r_signed, r_len=r_len)

    logging.info(f"Created 2D signed parameterization in a ({ns}, {nr}) grid")

    return s_index, r_index, centroid, u_vecs, r_edges

def parameterize_bundle(bundle, centroid, u_vecs, r_edges):
    """
    Parametrize a subject bundle onto atlas (s, r) grid,
    and mask bins with too few points as invalid.

    Parameters
    ----------
    bundle : ArraySequence
        Subject bundle
    centroid :  np.ndarray, shape (ns, 3)
        Atlas centroid coordinates
    uvecs :  np.ndarray, shape (ns, 3)
        Radial vectors for each along-tract segment
    r_edges:  np.ndarray, shape (nr+1, )
        Edges of radial bins

    Returns
    -------
    s_index : np.ndarray, shape (n_points,)
        Along-tract segment indices
    r_index : np.ndarray, shape (n_points,)
        Radial bin indices
    s_dist : np.ndarray, shape (n_points, )
        Distance to the closest point on the centroid
    valid_mask : np.ndarray, shape (n_points,)
        Boolean mask marking valid points. Points are discarded 
        if they are outside of the atlas radial range or if the
        bins they are in contain too few points.
    counts : np.ndarray, shape (ns, nr)
        Number of streamline points per bin
    """
    points = bundle.get_data()

    # Get along-tract segments
    s_dist, s_index = cKDTree(centroid).query(points, k=1)
    ns = len(centroid)

    # Get radial bins
    delta = points - centroid[s_index]
    r_signed = np.sum(delta * u_vecs[s_index], axis=1)
    r_index = np.digitize(r_signed, r_edges) - 1
    nr = len(r_edges) - 1

    # Clip points outside of the atlas radial range
    valid_mask = (r_index >= 0) & (r_index < nr)
    r_index = np.clip(r_index, 0, nr - 1)

    # Mask sparse bins based on point counts
    bin_ids = s_index * nr + r_index
    counts = np.bincount(bin_ids, minlength=ns * nr)
    thresh = np.clip(np.median(counts) * MASK_THRESHOLD, 50, 200)
    logging.info(f"Threshold for points per bin set to {thresh}")
    sparse_bins = np.where(counts < thresh)[0]
    if len(sparse_bins) > 0:
        mask_sparse = np.isin(bin_ids, sparse_bins)
        valid_mask[mask_sparse] = False

    counts = counts.reshape((ns, nr))

    return s_index, r_index, s_dist, valid_mask, counts

def prepare_bundle_pairs(org_bundle, rec_bundle, use_even_points=True):
    """
    TPA and BUAN profiling requires bundle in the native space and MNI space
    Resample both bundle such that they have the same number of points evenly distributed
    per streamline for creating segments. If both bundles have the same number of points and
    `use_even_points=False`, resampling is not performed.

    Parameters
    ----------
    org_bundle : ArraySequence 
        Streamlines in the native space
    rec_bundle : ArraySequence 
        Streamlines in the MNI space
    use_even_points : bool
        Resample to evenly distributed points for each streamline

    Returns
    -------
    org_bundle_rsp : ArraySequence
        Streamlines in the native space, resampled
    rec_bundle_rsp : ArraySequence
        Streamlines in the MNI space, resample to match org_bundle
    """
    if len(org_bundle) != len(rec_bundle):
        raise ValueError("Both bundles must have the same number of streamlines")

    if not use_even_points:
        if len(org_bundle.get_data()) == len(rec_bundle.get_data()):
            return org_bundle, rec_bundle

    n_lines = len(org_bundle)
    n_points = [len(l) for l in org_bundle] # the number of points per line in org_bundle

    org_bundle_rsp = [set_number_of_points(org_bundle[i], nb_points=n_points[i]) for i in range(n_lines)]
    rec_bundle_rsp = [set_number_of_points(rec_bundle[i], nb_points=n_points[i]) for i in range(n_lines)]

    return ArraySequence(org_bundle_rsp), ArraySequence(rec_bundle_rsp)


def buan_profile(org_bundle, s_index, s_dist, ns,
                 scalar_map, scalar_affine, 
                 epsilon=1e-8, seg_counts=None):
    """
    Compute along-tract scalar profile given precomputed indices using
    BUAN weighted mean.

    Parameters
    ----------
    org_bundle : ArraySequence
        Bundle in the same space as scalar_map
    s_index : np.ndarray, shape (n_points,)
        Point-wise along-tract segment indices
    s_dist : np.ndarray, shape (n_points,)
        Poinst-wise distance to the closest centroid point.
        The inverse distances are used as weights
    ns: int
        The number of segments
    scalar_map : np.ndarray
        Array of scalar map values to project to org_bundle.
        They must be in the same space
    scalar_affine : np.ndarray
        Affine matrix of scalar map
    epsilon : float
        Prevent division by zero during weight calculation
    seg_counts : np.ndarray, shape (ns,), optional
        Number of valid points per along-tract segment for masking 
        sparse segments. If None, no point-count masking is applied.

    Returns
    -------
    metric_profile : (ns, ) float
        Along-tract profile of scalar map value
    """  
    # Compute weights (based on distance) and segments indices
    weights = 1 / (s_dist + epsilon)

    if seg_counts is not None:
        nonzero_counts = seg_counts[seg_counts > 0]
        seg_thresh =  np.clip(np.median(nonzero_counts) * MASK_THRESHOLD, 50, 200) if len(nonzero_counts) > 0 else 0
        logging.debug(f"buan_profile: segment point threshold = {seg_thresh:.1f}")
    else:
        seg_thresh = None
    
    # Project scalar map to bundle
    metrics = values_from_volume(scalar_map, org_bundle, scalar_affine)
    metrics_flatten = np.array([m for metric_sl in metrics for m in metric_sl])
  
    # Compute weighted mean
    metric_profile = []
    for i in range(ns):
        # Mask sparse segments if seg_counts provided
        if seg_thresh is not None and seg_counts[i] < seg_thresh:
            metric_profile.append(0)
            continue
        segment_mask = (s_index==i) & (~np.isnan(metrics_flatten))
        if len(segment_mask)>1:
            vals = metrics_flatten[segment_mask]
            wts = weights[segment_mask]
            wts /= np.sum(wts)
            weighted_mean = np.sum(wts * vals)
            metric_profile.append(weighted_mean)
        else:
            metric_profile.append(0)
    return np.array(metric_profile)

def grid_profile(org_bundle, s_index, r_index, valid_mask,
                 ns, nr, scalar_map, scalar_affine):
    """
    Compute scalar profile per (s,r) grid cell given precomputed indices.

    Parameters
    ----------
    org_bundle : ArraySequence
        Bundle in the same space as scalar_map
    s_index : np.ndarray, shape (n_points,)
        Along-tract segment indices
    r_index : np.ndarray, shape (n_points,)
        Radial bin indices
    valid_mask : np.ndarray, shape (n_points,)
        Mask indicating which points to include
    ns: int
        The number of along-tract segments
    nr: int
        The number of radial bins
    scalar_map : np.ndarray
        Array of scalar map values to project to org_bundle.
        They must be in the same space
    scalar_map : np.ndarray
        Affine matrix of scalar map

    Returns
    -------
    metric_grid : (ns, ) float
        Along-tract profile of scalar map value
    """    
    # Project scalar map to bundle
    metrics = values_from_volume(scalar_map, org_bundle, scalar_affine)
    metrics_flatten = np.array([val for values_line in metrics for val in values_line])
    
    # Apply mask
    s_index_masked = s_index[valid_mask]
    r_index_masked = r_index[valid_mask]
    metrics_masked = metrics_flatten[valid_mask]

    # Flatten 2D indices to 1D
    bin_ids = s_index_masked * nr + r_index_masked

    # Sum FA in each bin
    metric_sum = np.bincount(bin_ids, weights=metrics_masked, minlength=ns*nr)
    counts = np.bincount(bin_ids, minlength=ns * nr)

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        metric_mean = metric_sum / counts
        metric_mean[counts == 0] = 0

    # Reshape back to 2D grid
    metric_grid = metric_mean.reshape((ns, nr))

    return metric_grid