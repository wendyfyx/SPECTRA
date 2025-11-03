import logging

import numpy as np
from scipy.spatial import cKDTree

from nibabel.streamlines.array_sequence import ArraySequence
from dipy.tracking.streamline import set_number_of_points, Streamlines, values_from_volume
from dipy.tracking.utils import length
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.clustering import QuickBundles


def get_n_segment_by_length(ref_bundle, segment_length):
    '''Calculate the number of segments to produce given segment length (mm)'''
    avg_length = np.mean(list(length(ref_bundle)))
    return max(int(np.round(avg_length / segment_length)), 1)
