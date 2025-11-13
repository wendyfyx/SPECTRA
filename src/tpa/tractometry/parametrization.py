import logging

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from nibabel.streamlines.array_sequence import ArraySequence
from dipy.tracking.utils import length
from dipy.tracking.streamline import set_number_of_points, Streamlines, \
                                     values_from_volume, orient_by_streamline
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundles


def get_centroid(bundle, n_points=50, thresh=50):
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

def create_radial_bins(r_signed, r_len=5):
    """
    Bin signed radial distances based on bin length.
    Merge bins on the edge if they have too few points.

    Parameters
    ----------
    r_signed : np.ndarray, shape (M,)
        Signed radial distances for each point
    r_len : float
        Target radial bin size

    Returns
    -------
    r_index : np.ndarray, shape (M,)
        Bin index for each point
    nr : int
        Total number of bins after merging
    """
    r_min, r_max = r_signed.min(), r_signed.max()
    nr = int(np.ceil((r_max - r_min) / r_len))
    r_edges = np.linspace(r_min, r_max, nr + 1)

    # Create r_index
    r_index = np.digitize(r_signed, r_edges) - 1
    r_index = np.clip(r_index, 0, nr - 1)

    # Define point threshold as median point count across bins * 0.5
    point_count = np.bincount(r_index, minlength=nr)
    thresh = np.median(point_count) * 0.5

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
    
def get_grid_from_atlas(atlas_bundle, s_len=5, r_len=5):
    """
    Define shared (s, r) grid based on atlas bundle geometry
    along tract and radially across tract.

    Returns
    -------
    s_index : (n_points, )
        Point-wise assignment of along-tract segments
    r_index : int
        Point-wise assignment of radial bins
    centroid : (ns, 3)
        Atlas centroid coordinates
    uvecs : (ns, 3)
        Radial vectors for each along-tract segment
    r_edges: (nr+1, )
        Edges of radial bins
    """

    # Create along-tract segments
    ns = get_n_segment_by_length(atlas_bundle, s_len=s_len)
    centroid = get_centroid(atlas_bundle, n_points=ns, thresh=100)
    points = atlas_bundle.get_data()
    _, s_index = cKDTree(centroid).query(points, k=1)

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

    logging.info(f"Created 2D signed parametrization in a ({ns}, {nr}) grid")

    return s_index, r_index, centroid, u_vecs, r_edges

def parametrize_bundle(bundle, centroid, u_vecs, r_edges):
    """
    Parametrize a subject bundle onto atlas (s, r) grid,
    and mark bins with too few points as invalid.

    Parameters
    ----------
    bundle : ArraySequence
        Subject bundle
    centroid : (ns, 3)
        Atlas centroid points
    u_vecs : (ns, 3)
        Tangent vectors along centroid
    r_edges : (nr+1,)
        Radial bin edges

    Returns
    -------
    s_index : (n_points,)
        Along-tract segment indices
    r_index : (n_points,)
        Radial bin indices
    s_dist : (n_points, )
        Distance to the closest point on the centroid
    valid_mask : (n_points,)
        Boolean mask marking valid points. Points are discarded 
        if they are outside of the atlas radial rage or if the
        bins they are in contain too few points.
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
    valid_mask = (r_index >= 0) & (r_index < len(r_edges) - 1)
    r_index = np.clip(r_index, 0, nr - 1)

    # Mask sparse bins based on point counts
    bin_ids = s_index * nr + r_index
    counts = np.bincount(bin_ids, minlength=ns * nr)
    thresh = np.clip(np.median(counts) * 0.5, 50, 200)
    print(thresh)
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
    org_bundle : (ArraySequence) Streamlines in the native space
    rec_bundle : (ArraySequence) Streamlines in the MNI space
    use_even_points : (bool) Resample to evenly distributed points for each streamline

    Returns
    -------
    org_bundle_rsp : (ArraySequence) Streamlines in the native space, resampled
    rec_bundle_rsp : (ArraySequence) Streamlines in the MNI space, resample to match org_bundle
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


def buan_profile(org_bundle, s_index, s_dist,
                 scalar_map, scalar_affine, 
                 epsilon=1e-8):
    """
    Compute along-tract scalar profile given precomputed indices using
    BUAN weighted mean.

    Parameters
    ----------
    org_bundle : ArraySequence
        Bundle in the same space as scalar_map
    s_index : (N,) int
        Point-wise along-tract segment indices
    s_dist : (N,) float
        Poinst-wise distance to the closest centroid point
    scalar_map : np.ndarray
        Array of scalar map values to project to org_bundle.
        They must be in the same space
    scalar_map : np.ndarray
        Affine matrix of scalar map
    epsilon : float
        Prevent division by zero during weight calculation

    Returns
    -------
    metric_profile : (ns, ) float
        Along-tract profile of scalar map value
    """  
    # Compute weights (based on distance) and segments indices
    weights = 1 / (s_dist + epsilon)
    ns = s_index.max() + 1
    
    # Project scalar map to bundle
    metrics = values_from_volume(scalar_map, org_bundle, scalar_affine)
    metrics_flatten = np.array([m for metric_sl in metrics for m in metric_sl])
  
    # Compute weighted mean
    metric_profile = []
    for i in range(ns):
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
# 
def grid_profile(org_bundle, s_index, r_index, valid_mask,
                scalar_map, scalar_affine):
    """
    Compute scalar profile per (s,r) grid cell given precomputed indices.

    Parameters
    ----------
    org_bundle : ArraySequence
        Bundle in the same space as scalar_map
    s_index : (N,) int
        Along-tract segment indices
    r_index : (N,) int
        Radial bin indices
    valid_mask : (N,) bool
        Mask indicating which points to include
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

    ns = s_index_masked.max() + 1
    nr = r_index_masked.max() + 1

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