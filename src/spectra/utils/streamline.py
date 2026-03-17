### From Iyad, adapted from MeTA https://github.com/USC-LoBeS/meta/blob/cf8d001824616bfd5220657f9f08c8948994a562/meta/io/streamline.py

import io
import sys
import gzip
import logging
import argparse
import numpy as np
import scipy.io as sio
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import TrkFile
from nibabel.streamlines.tck import TckFile
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import transform_streamlines


def parse_tt(tinytrack):
    """
    Parse DSI-Studio TinyTrack format (.tt.gz) to extract streamlines and metadata.

    Parameters:
        tinytrack: Path to the TinyTrack file.

    Returns:
        streamlines: List of streamlines in LPS space.
        tt_affine: Affine transformation matrix.
        dimension: Dimensions of image.
        voxel_size: Voxel size of image.
        voxel_order: Voxel order of image.
    """

    if tinytrack.endswith('.tt.gz'):
        with gzip.open(tinytrack, 'rb') as f:
            data = f.read()
        mat = sio.loadmat(io.BytesIO(data), appendmat=False, spmatrix=False)
    tt_affine = mat['trans_to_mni'].reshape(4,4)
    dimension = tuple(mat['dimension'].ravel().astype(int))
    voxel_size = tuple(mat['voxel_size'].ravel().astype(float))
    voxel_order = "".join(nib.aff2axcodes(tt_affine))

    buf1 = mat.get('track').flatten()
    buf2 = buf1.view(np.int8)
    length = len(buf1)
    pos = []
    i = 0
    while i < length:
        pos.append(i)
        track_length = np.frombuffer(buf1[i:i+4].tobytes(), dtype=np.uint32)[0]
        i += int(track_length) + 13

    streamlines = []
    for p in pos:
        size_val = np.frombuffer(buf1[p:p+4].tobytes(), dtype=np.uint32)[0]
        num_points = int(size_val // 3)
        x = np.frombuffer(buf1[p+4:p+8].tobytes(), dtype=np.int32)[0]
        y = np.frombuffer(buf1[p+8:p+12].tobytes(), dtype=np.int32)[0]
        z = np.frombuffer(buf1[p+12:p+16].tobytes(), dtype=np.int32)[0]

        track_pts = np.empty((num_points, 3), dtype=np.float32)
        track_pts[0, :] = [x, y, z]
        p_offset = p + 16

        for j in range(1, num_points):
            dx = int(buf2[p_offset])
            dy = int(buf2[p_offset + 1])
            dz = int(buf2[p_offset + 2])
            x += dx; y += dy; z += dz
            track_pts[j, :] = [x, y, z]
            p_offset += 3
        track_pts /= 32.0
        streamlines.append(track_pts)

    logging.debug(f"Parsed {len(streamlines)} streamlines in total")
    logging.debug(f"Affine: \n {tt_affine},\n Dimension: {dimension},\n Voxel Size: {voxel_size},\n Voxel Order: {voxel_order}")
    return streamlines, tt_affine, dimension, voxel_size, voxel_order


def read_streamlines(bundle_path, reference=None, transform=None):
    """
    Read streamlines from a bundle file, supporting TRK, TCK, and TinyTrack formats.
    
    Parameters:
        bundle_path: Path to segmented white matter bundle file.

    Returns:
        streamlines: List of streamlines in RASMM space.
    """
    if bundle_path.endswith('.tck'):
        bundle = load_tractogram(bundle_path, reference, bbox_valid_check=False)
        streamlines = bundle.streamlines
        if transform is not None:
            streamlines = transform_streamlines(streamlines, transform)

        groups = bundle.groups if hasattr(bundle, 'groups') else {}
        affine = bundle.affine if hasattr(bundle, 'affine') else np.eye(4)
        dimension = bundle.dimensions if hasattr(bundle, 'dimensions') else None
        return streamlines, groups, affine, dimension

    if bundle_path.endswith(('.trk', '.trx')):
        bundle =  load_tractogram(bundle_path, "same", bbox_valid_check=False)
        streamlines = bundle.streamlines
        if transform is not None:
            streamlines = transform_streamlines(streamlines, transform)

        groups = bundle.groups if hasattr(bundle, 'groups') else {}
        affine = bundle.affine if hasattr(bundle, 'affine') else np.eye(4)
        dimension = bundle.dimensions if hasattr(bundle, 'dimensions') else None
        return streamlines, groups, affine, dimension

    if bundle_path.endswith('.tt.gz'):
        ## Read DSI-Studio format (TinyTrack):
        streamlines, tt_affine, dimension, _, _ = parse_tt(bundle_path)
        streamlines_lps_center = [s - 0.5 for s in streamlines]
        streamlines = transform_streamlines(streamlines_lps_center, tt_affine)
        if transform is not None:
            streamlines = transform_streamlines(streamlines, transform)

        groups = {}
        affine = tt_affine
        dimension = dimension
        return ArraySequence(streamlines), groups, affine, dimension
    
    raise ValueError(f"Only .trk, .tck, .tt.gz are supported")


def convert_tinytrack_to_trk_tck(bundle_path, output_path):
    """
    Convert TinyTrack (.tt.gz) to TRK or TCK format.

    Args:
        bundle_path: Path to segmented white matter bundle file in TinyTrack format.
        output_path: Output paths, ends either in 'trk' or 'tck'.
    """

    # if bundle_path.endswith('.tt.gz'):
    #     output = bundle_path[:-6]
    # else:
    #     output = bundle_path

    ## Read TinyTrack:
    streamlines, tt_affine, dimension, voxel_size, voxel_order = parse_tt(bundle_path)
    streamlines_lps_center = [s - 0.5 for s in streamlines]
    tractogram = Tractogram(streamlines_lps_center, affine_to_rasmm=tt_affine)

    if output_path.endswith == 'tck':
        tck = TckFile(tractogram=tractogram)
        tck.save(output_path)
    else:
        header = {}
        header["dimensions"] = dimension
        header["voxel_sizes"] = voxel_size
        header["voxel_to_rasmm"] = tt_affine
        header["voxel_order"] = voxel_order
        header["magic_number"] = b"TRACK"
        header["nb_streamlines"] = len(streamlines_lps_center)
        header["version"] = 2
        header["hdr_size"] = 1000
        trk = TrkFile(tractogram, header=header)
        trk.save(output_path) 

# def load_streamlines(trk_file):
#     '''Helper function for loading streamlines'''
#     tractogram = load_trk(trk_file, reference='same', bbox_valid_check=False)
#     return tractogram.streamlines, tractogram.affine


def save_streamlines(streamlines, out_file, ref_file):
    '''Save streamlines to file given a reference file'''
    sft = StatefulTractogram(streamlines, ref_file, Space.RASMM)
    sft.remove_invalid_streamlines()
    save_trk(sft, out_file, bbox_valid_check=False)
    logging.info(f'Saved {len(streamlines)} streamlines to {out_file}')