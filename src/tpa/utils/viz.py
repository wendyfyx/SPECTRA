import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go

def alternating_colors(s_index=None, r_index=None, pattern='viridis', valid_mask=None):
    """
    Generate RGB colors for points based on optional s_index and r_index,
    with invalid points marked as gray.
    
    Parameters
    ----------
    s_index : array-like, shape (N,), optional
        Axial segment indices for each point.
    r_index : array-like, shape (N,), optional
        Radial bin indices for each point.
    pattern : str, list of 2 colors, or list of list of 2 colors
        - If list of 2 colors (e.g., ['red','green']), used as 1D checkerboard along s or r.
        - If list of list of 2 colors (e.g., [[a,b],[c,d]]), used as 2D checkerboard along s and r.
    valid_mask : array-like, shape (N,), optional
        1 for valid points, 0 for invalid points (marked gray).
        
    Returns
    -------
    colors : list of tuple
        List of RGB tuples for each point.
    """
    
    # Determine number of points
    if s_index is not None:
        s_index = np.asarray(s_index)
        N = len(s_index)
    elif r_index is not None:
        r_index = np.asarray(r_index)
        N = len(r_index)
    else:
        return [tuple(mpl.colors.to_rgb('gray'))]
    
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if len(valid_mask) != N:
            raise ValueError("valid_mask must have the same length as s_index or r_index")
    else:
        valid_mask = np.ones(N, dtype=bool)
    
    colors = np.zeros((N,3))
    gray = np.array(mpl.colors.to_rgb('gray'))
    
    # Case 2D checkerboard
    if s_index is not None and r_index is not None and isinstance(pattern, (list, tuple)) and all(isinstance(p, (list, tuple)) for p in pattern):
        r_index = np.asarray(r_index)
        for i in range(N):
            if valid_mask[i]:
                row = pattern[s_index[i] % len(pattern)]
                c = row[r_index[i] % len(row)]
                colors[i] = mpl.colors.to_rgb(c)
            else:
                colors[i] = gray
                
    # Case 1D checkerboard
    elif (s_index is not None or r_index is not None) and isinstance(pattern, (list, tuple)) and len(pattern) == 2:
        idx_array = s_index if s_index is not None else r_index
        for i in range(N):
            colors[i] = mpl.colors.to_rgb(pattern[idx_array[i] % 2]) if valid_mask[i] else gray

    return [tuple(c) for c in colors]


def plot_uvecs(centroids, r_vecs, r_len=5):
    """
    Plot the centroid and signed radial vectors (+r and -r) along the bundle using Plotly.
    """
    fig = go.Figure()

    centroids = [centroids] if centroids.ndim == 2 else centroids
    r_vecs = [r_vecs] if r_vecs.ndim == 2 else r_vecs

    for i, (centroid, r_vec) in enumerate(zip(centroids, r_vecs)):

        ns = len(centroid)
        r_pos = centroid + r_vec * r_len
        r_neg = centroid - r_vec * r_len
        
    
        # Centroid line
        fig.add_trace(go.Scatter3d(
            x=centroid[:,0], y=centroid[:,1], z=centroid[:,2],
            mode='lines+markers',
            line=dict(color='black', width=4),
            marker=dict(size=3),
            name='Centroid'
        ))
    
        # +r arrows
        for i in range(ns):
            fig.add_trace(go.Scatter3d(
                x=[centroid[i,0], r_pos[i,0]],
                y=[centroid[i,1], r_pos[i,1]],
                z=[centroid[i,2], r_pos[i,2]],
                mode='lines',
                line=dict(color='blue', width=3),
                name='+r' if i==0 else None,  # only label first
                showlegend=i==0
            ))
        
        # -r arrows
        for i in range(ns):
            fig.add_trace(go.Scatter3d(
                x=[centroid[i,0], r_neg[i,0]],
                y=[centroid[i,1], r_neg[i,1]],
                z=[centroid[i,2], r_neg[i,2]],
                mode='lines',
                line=dict(color='red', width=3),
                name='-r' if i==0 else None,
                showlegend=i==0
            ))
        
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        title="Bundle centroid and signed radial vectors (+r:blue, -r:red)"
    )
    
    fig.show()