import numpy as np
import plotly.graph_objects as go
import matplotlib as mpl

from tpa.viz.colors import directional_color, colors_from_values

def get_cam_settings_plotly(cam_name='Axial'):
    EYE_SCALE=2
    CAM_SETTINGS = {
        'Axial': {'up': dict(x=0, y=1, z=0), 'center': dict(x=0, y=0, z=0), 'eye': dict(x=0, y=0, z=EYE_SCALE) },
        'Sagittal_L': {'up': dict(x=0, y=0, z=1), 'center': dict(x=0, y=0, z=0),'eye': dict(x=-1*EYE_SCALE, y=0, z=0) },
        'Sagittal_R': {'up': dict(x=0, y=0, z=1), 'center': dict(x=0, y=0, z=0),'eye': dict(x=EYE_SCALE, y=0, z=0) },
        'Coronal_A': {'up': dict(x=0, y=0, z=1), 'center': dict(x=0, y=0, z=0), 'eye': dict(x=0, y=EYE_SCALE, z=0)},
        'Coronal_P': {'up': dict(x=0, y=0, z=1), 'center': dict(x=0, y=0, z=0), 'eye': dict(x=0, y=-1*EYE_SCALE, z=0)}
        }
    return CAM_SETTINGS[cam_name]


def get_line_trace_multi(lines, fake_tube=True, opacity=1, linecolors=None, **kwargs):
    if linecolors is None:
        colors = directional_color(lines)
    elif isinstance(linecolors, str):
        colors = [linecolors]*len(lines)
    else:
        colors = [colors_from_values(vals, plot_cmap=i==0, **kwargs) for i, vals in enumerate(linecolors)]
    traces = []
    for i, stream in enumerate(lines):
        points = np.array(stream)
        if fake_tube:
            trace_border = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='lines', 
                                        opacity=opacity,
                                        line=dict(color='black', width=4), showlegend=False)
            traces.append(trace_border)
        trace = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='lines', 
                             opacity=opacity,
                             line=dict(color=colors[i], width=3), showlegend=False)
        traces.append(trace)
    return traces

def get_line_trace(line, markers=False, color='black'):
    mode = 'lines+markers' if markers else 'lines'
    return go.Scatter3d(
            x=line[:,0], y=line[:,1], z=line[:,2],
            mode=mode,
            line=dict(color=color, width=4),
            marker=dict(size=3),
        )

def plot_grids(
    centroids, 
    u_vecs, 
    r_len=5, 
    streamlines=None, 
    n_bins_pos=None, 
    n_bins_neg=None, 
    cam='Axial',
    **kwargs
):
    """
    Plot the centroid line, the +- radial vectors at each point,
    and N radial bin lines parallel to the centroid.

    Parameters
    ----------
    centroids  : (ns, 3) or (n_bundles, ns, 3)
    u_vecs     : (ns, 3) or (n_bundles, ns, 3)
    r_len      : float, distance per radial bin
    n_bins_pos : int, number of bins on the positive side
    n_bins_neg : int, number of bins on the negative side
    cam        : str, view direction
    """
        
    fig = go.Figure()

    centroids = [centroids] if centroids.ndim == 2 else centroids
    u_vecs    = [u_vecs] if u_vecs.ndim == 2 else u_vecs

    for centroid, r_vec in zip(centroids, u_vecs):

        ns = len(centroid)

        # Centroid line
        fig.add_trace(go.Scatter3d(
            x=centroid[:,0], y=centroid[:,1], z=centroid[:,2],
            mode='lines+markers',
            line=dict(color='black', width=6),
            marker=dict(size=4),
            name='Centroid'
        ))

        # Radial vectors at each s-position (+r and -r)
        if n_bins_pos is not None and n_bins_neg is not None:
            r_len_pos = r_len * n_bins_pos
            r_len_neg = r_len * n_bins_neg
        else:
            r_len_pos = r_len 
            r_len_neg = r_len
        r_pos = centroid + r_vec * r_len_pos
        r_neg = centroid - r_vec * r_len_neg

        for i in range(ns):
            fig.add_trace(go.Scatter3d(
                x=[centroid[i,0], r_pos[i,0]],
                y=[centroid[i,1], r_pos[i,1]],
                z=[centroid[i,2], r_pos[i,2]],
                mode='lines',
                line=dict(color='blue', width=3),
                name='+r' if i == 0 else None,
                showlegend=i == 0
            ))

            fig.add_trace(go.Scatter3d(
                x=[centroid[i,0], r_neg[i,0]],
                y=[centroid[i,1], r_neg[i,1]],
                z=[centroid[i,2], r_neg[i,2]],
                mode='lines',
                line=dict(color='red', width=3),
                name='-r' if i == 0 else None,
                showlegend=i == 0
            ))

        # Radial bin lines: full curves parallel to the centroid
        if n_bins_pos is not None:
            for k in range(1, n_bins_pos+1):
                # + bins
                pos_bin = centroid + k * r_vec * r_len
                fig.add_trace(go.Scatter3d(
                    x=pos_bin[:,0], y=pos_bin[:,1], z=pos_bin[:,2],
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name=f'+{k} bin' if k == 1 else None,
                    showlegend=False
                ))

        if n_bins_neg is not None:
            for k in range(1, n_bins_neg+1):
                # - bins
                neg_bin = centroid - k * r_vec * r_len
                fig.add_trace(go.Scatter3d(
                    x=neg_bin[:,0], y=neg_bin[:,1], z=neg_bin[:,2],
                    mode='lines',
                    line=dict(color='red', width=3),
                    name=f'-{k} bin' if k == 1 else None,
                    showlegend=False
                ))

    if streamlines is not None:
        for i in get_line_trace_multi(streamlines, opacity=0.7, fake_tube=False, **kwargs):
            fig.add_trace(i)

    # Layout
    camera = get_cam_settings_plotly(cam)
    fig.update_layout(
        width=600, height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            zaxis=dict(showgrid=False, visible=False),
            camera=camera, aspectmode='data'
        ),
        showlegend=True,
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode=False,
        font=dict(size=20)
    )
    
    fig.show()