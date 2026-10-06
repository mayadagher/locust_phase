'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
import os
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from tqdm import tqdm
import gc

from helper_fns import *
from data_handling import load_cluster_stats_h5
from cluster_analysis import find_reflections

'''_____________________________________________________HTML FUNCTIONS____________________________________________________________'''

def interactive_voronoi_overlay(ds:xr.Dataset, param:str, output_dir:str, arena_center: np.ndarray, arena_radius: float, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1, n_bins:int = 10, cmap:str = 'viridis', fs:float = 5, bb:bool = False):

    # Define position array to save time from accessing ds
    if bb:
        x = ds['x_raw'].transpose('frame', 'id')
        y = ds['y_raw'].transpose('frame', 'id')
        positions = np.stack([x, y])

        z = ds[param].transpose('frame', 'id').values # (n_frames, max_ids)
    else:
        positions = np.stack([ds['centroid_x'], ds['centroid_y']]) # (2, n_frames, max_ids)
        z = ds[param].values # (n_frames, max_ids)

    # Set up colour bins
    colourscale = pc.sample_colorscale(cmap, n_bins)
    z_min, z_max = np.nanmin(z), np.nanmax(z)
    print('z extrema: ', z_min, z_max)
    
    if z_max > 5*np.nanstd(z) + np.nanmean(z): # If maximum is a huge outlier, replace with more reasonable maximum
        z_max = np.nanquantile(z, 0.95)
        print('z 95th quantile: ', z_max)
    bins = np.linspace(z_min, z_max, n_bins + 1)

    # Filter out detections outside of arena
    dist_from_center = np.sqrt((positions[0,:,:] - arena_center[0])**2 + (positions[1,:,:] - arena_center[1])**2) # (n_frames, max_ids)
    outside_arena_mask = dist_from_center > arena_radius

    # Define valid mask
    valid_mask = (~np.isnan(positions).any(axis = 0)) & (~outside_arena_mask) # (n_frames, max_ids)

    # Get frames
    abs_frames, ds_idcs = get_frame_slice(ds, start_frame, end_frame, subsample)

    fig = go.Figure()

    # Track how many traces belong to each frame
    traces_per_frame = []
    areas = []

    for f in ds_idcs:
        # Filter valid positions and z values
        valid_positions_t = positions[:, f, valid_mask[f]]  # (2, n_ids)
        valid_z_t = z[f, valid_mask[f]]

        if valid_positions_t.shape[1] < 3:
            traces_per_frame.append(0)
            continue

        # Compute voronoi tessellation
        vor = Voronoi(valid_positions_t.T)

        # One set of polygons per colour bin
        bin_xs = [[] for _ in range(n_bins)]
        bin_ys = [[] for _ in range(n_bins)]

        # Separate polygons for NaN values
        nan_xs = []
        nan_ys = []

        # Iterate over each point
        for point_idx, z_val in enumerate(valid_z_t):
            region_idx = vor.point_region[point_idx]
            region = vor.regions[region_idx]

            # Exclude regions with no points
            if len(region) == 0:
                continue

            # Clip and order vertices
            vertices = vor.vertices[
                np.array(region)[np.array(region) != -1].astype(int)
            ]
            poly = Polygon(
                clip_voronoi_region(
                    vertices,
                    arena_center,
                    arena_radius,
                    0.05
                )
            )

            if poly.is_empty:
                continue

            areas.append(poly.area)

            pos = np.array(poly.exterior.coords)
            xs = pos[:, 0].tolist() + [None]
            ys = pos[:, 1].tolist() + [None]

            if np.isnan(z_val):
                # Keep polygon geometry but don't fill it
                nan_xs.extend(xs)
                nan_ys.extend(ys)
            else:
                bin_idx = np.searchsorted(bins, z_val, side='right') - 1
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)

                bin_xs[bin_idx].extend(xs)
                bin_ys[bin_idx].extend(ys)

        # Add coloured polygons
        for b in range(n_bins):
            fig.add_trace(go.Scatter(
                x=bin_xs[b],
                y=bin_ys[b],
                fill='toself',
                mode='lines',
                fillcolor=colourscale[b],
                line=dict(width=0.5, color='rgba(0,0,0,0.3)'),
                visible=False,
                showlegend=False
            ))

        # Add NaN polygons with transparent fill
        fig.add_trace(go.Scatter(
            x=nan_xs,
            y=nan_ys,
            fill='toself',
            mode='lines',
            fillcolor='rgba(0,0,0,0)',
            line=dict(width=0.5, color='rgba(0,0,0,0.3)'),
            visible=False,
            showlegend=False
        ))

        traces_per_frame.append(n_bins + 1)

    # Add arena outline
    theta = np.linspace(0, 2*np.pi, 300)
    fig.add_trace(go.Scatter(x=arena_center[0] + arena_radius*np.cos(theta), y=arena_center[1] + arena_radius*np.sin(theta), mode='lines', line=dict(color='black'), showlegend=False))
    circle_trace_idx = len(fig.data) - 1

    # Add dummy scatter points for colour bar
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(colorscale=cmap, cmin=z_min, cmax=z_max, color=[z_min], colorbar=dict(title=dict(text=param, side='right'),
                tickvals=np.linspace(z_min, z_max, 6).tolist(), ticktext=[f'{v:.2f}' for v in np.linspace(z_min, z_max, 6)], thickness=20, len=0.75), showscale=True), showlegend=False, visible=True))
    colourbar_trace_idx = len(fig.data) - 1

    # Build slider steps — each step makes exactly its frame's traces visible
    slider_steps = []
    total_traces = sum(traces_per_frame) + 2 # Add 2 for circle and colour bar trace
    cumulative = 0

    for i, count in enumerate(traces_per_frame):
        visibility = [False] * total_traces
        for j in range(cumulative, cumulative + count):
            visibility[j] = True
        visibility[circle_trace_idx] = True  # always show circle
        visibility[colourbar_trace_idx] = True

        slider_steps.append(dict(
            method='restyle',
            args=[{'visible': visibility}],
            label=str(abs_frames[i]),  # display actual frame number
        ))
        cumulative += count

    # Make first frame visible by default
    if traces_per_frame[0] > 0:
        for i in range(traces_per_frame[0]):
            fig.data[i].visible = True

    # Add slider and fix axes
    fig.update_layout(sliders=[dict(active=0, steps=slider_steps, currentvalue=dict(prefix='Frame: ', visible=True), pad=dict(t=50))],
                      xaxis=dict(range=[arena_center[0] - arena_radius * 1.1, arena_center[0] + arena_radius * 1.1], constrain='domain'),
                      yaxis=dict(range=[arena_center[1] - arena_radius * 1.1, arena_center[1] + arena_radius * 1.1],
                      scaleanchor='x', scaleratio=1, constrain='domain'),
                      title='Voronoi Tessellation Over Time')
    

    output_path = f"{output_dir}voronoi_sliders/voronoi_slider_{param}_{abs_frames[0]}_{abs_frames[-1]}_fs_{fs/round(np.diff(abs_frames)[0])}.html"
    fig.write_html(output_path)
    print(f"Saved to {output_path}.")

    return

def interactive_voronoi_distributions(ds:xr.Dataset, output_dir:str, arena_center:np.ndarray, arena_radius:float, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1, n_bins: list[int] = [40, None, 30], fs:float = 5, density_factor:float = 1):

    # Define position array to save time from accessing ds
    positions = np.stack([ds['centroid_x'], ds['centroid_y']]) # (2, n_frames, max_ids)
    densities = ds['density_voronoi_None'].values*density_factor # (n_frames, max_ids)

    # Filter out detections outside of arena
    dist_from_center = np.sqrt((positions[0,:,:] - arena_center[0])**2 + (positions[1,:,:] - arena_center[1])**2) # (n_frames, max_ids)
    outside_arena_mask = dist_from_center > arena_radius

    # Define valid mask
    valid_mask = (~np.isnan(positions).any(axis = 0)) & (~np.isnan(densities)) & (~outside_arena_mask) # (n_frames, max_ids)

    # Get frames
    abs_frames, ds_idcs = get_frame_slice(ds, start_frame, end_frame, subsample)

    # Name parameters
    titles = ['Voronoi areas', 'Voronoi neighbours', 'Voronoi densities']
    xlabels = ['Area (㎡)', 'Number of neighbours (n)', 'Density (n/㎡)']
    n_params = len(titles)

    fig = make_subplots(rows=1, cols= 3, subplot_titles=titles, horizontal_spacing=0.08, vertical_spacing=0.12)

    all_areas = []
    all_nbr_counts = []

    max_area = 0
    max_nbrs = 0
    max_density = np.quantile(densities[valid_mask], 0.999) # Some crazy outliers need to be excluded

    # Iterate over each frame to collect area and nbr count data
    for ds_idx in ds_idcs:

        # Filter valid positions and z values
        valid_positions_t = positions[:, ds_idx, valid_mask[ds_idx]].T # (n_ids, 2)

        if valid_positions_t.shape[0] < 3:
            continue

        # Compute voronoi tessellation
        vor = Voronoi(valid_positions_t)

        # Now we don't care about the order of the polygons so we can compute the polygons in one line
        polys = [Polygon(clip_voronoi_region(vor.vertices[np.array(region)[(np.array(region) != -1).astype(bool)].astype(int)], arena_center, arena_radius)) for region in vor.regions]

        # Get areas of each polygon
        areas = [poly.area/density_factor for poly in polys]

        # Compute neighbour relationships from Voronoi ridges
        nbrs = {i: set() for i in range(len(valid_positions_t))}
        for i1, i2 in vor.ridge_points:
            nbrs[i1].add(i2)
            nbrs[i2].add(i1)
        indcs = [sorted(list(v)) for v in nbrs.values()]
        num_nbrs = np.array([len(nbrs) for nbrs in indcs])

        # Append data to lists
        all_areas.append(areas)
        all_nbr_counts.append(num_nbrs)

        # Update maxima
        if np.max(areas) > max_area:
            max_area = np.max(areas)
        if np.max(num_nbrs) > max_nbrs:
            max_nbrs = np.max(num_nbrs)

    # Add ALL traces upfront (one per param per frame), only the first frame visible
    maxes = [max_area, max_nbrs, max_density]
    maxes_counts = [0, 0, 0]
    
    if not n_bins[1]: # If nbr bins not specified
        n_bins[1] = max_nbrs

    for f_idx, (frame_num, ds_idx) in enumerate(zip(abs_frames, ds_idcs)):
        # Iterate over params to get each histogram
        params = [all_areas[f_idx], all_nbr_counts[f_idx], densities[ds_idx, valid_mask[ds_idx]]]

        for i, param in enumerate(params):
            if i == 1:
                bin_edges = np.arange(0, max_nbrs + 1)
                counts, _ = np.histogram(param, bins = bin_edges, range = (0, maxes[i]))
            else:
                counts, bin_edges = np.histogram(param, bins=n_bins[i], range=(0, maxes[i]))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            if max(counts) > maxes_counts[i]:
                maxes_counts[i] = max(counts)

            fig.add_trace(go.Bar(x=bin_centers, y=counts,
                                 name=titles[i],
                                 showlegend=False,
                                 visible=(f_idx == 0),  # only first frame visible initially
                                 marker_color='steelblue',
                                 marker_line_width=0), row=1, col=i + 1)

    # Each slider step sets visibility: only the n_params traces for that frame are True
    total_traces = positions.shape[1] * n_params
    steps = []
    for f_idx, frame_num in enumerate(abs_frames):
        visibility = [False] * total_traces
        for i in range(n_params):
            visibility[f_idx * n_params + i] = True

        steps.append({
            'method': 'restyle',
            'label': str(frame_num),
            'args': [{'visible': visibility}],
        })

    sliders = [{
        'active': 0,
        'currentvalue': {'prefix': 'Frame: ', 'visible': True, 'xanchor': 'center'},
        'pad': {'t': 50},
        'steps': steps,
    }]

    fig.update_layout(
        sliders=sliders,
        bargap=0.02,
        title_text='Distributions by Frame')

    for i in range(n_params):
        fig.update_xaxes(range=[0, maxes[i]], row=1, col=i + 1, title_text = xlabels[i])
        fig.update_yaxes(range=[0, maxes_counts[i]], row=1, col=i + 1, title_text = 'Counts' if not i else '')

    output_path = os.path.join(output_dir, f'voronoi_sliders/vor_distributions_{abs_frames[0]}_{abs_frames[-1]}_fs_{fs/round(np.diff(abs_frames)[0])}.html')
    fig.write_html(output_path)
    print(f'Saved to {output_path}')
    return fig

def interactive_cluster_analysis(input_path: str, min_obs:int = 5, max_layers:int | None = None, n_bins:int = 21):

    # Load data
    data = load_cluster_stats_h5(input_path)
    
    # Relative frames in integers
    rel_frames = np.arange(-1*round((len(data) - 1)/2), round((len(data) - 1)/2) + 1)

    # Initialize extrema dictionaries and parameter strings
    all_params = data['0'].keys()
    max_x = {p: 0 for p in all_params}
    max_x['medPols'], max_x['meanThetas'], max_x['p_by_layer'], max_x['p_from_edge'] = 1, np.pi, 1, 1
    min_x = {p: 0 for p in all_params}
    min_x['ns'], min_x['areas'], min_x['meanThetas'] = 2, np.inf, -np.pi

    # Iterate over frames to collect maxima
    for rel_idx in rel_frames:
        
        for param in ['ns', 'areas', 'varPols', 'medDs', 'varDs', 'varThetas', 'd_by_layer', 'd_from_edge']:

            # Update maximum overall value
            vals = data[str(rel_idx)][param]

            if param == 'areas':
                min_val = np.nanquantile(vals, 0.01)
                if min_val < min_x[param]:
                    min_x[param] = min_val

            if type(vals) == dict: # If layers
                max_val = np.nanquantile(vals['data'], 0.999) # Ignoring crazy outliers
            else:
                max_val = np.nanquantile(vals, 0.999)

            if max_val > max_x[param]:
                max_x[param] = float(max_val)

        del vals

    titles = ['N distribution', 'Area distribution', 'Mean θ distribution', 'Area vs N', 'Med. pol & density vs N', 'Var. pol & density vs N', 'θ vs N', 'Pol vs density',
              'Polarization by layer (centre)', 'Polarization by layer (edge)', 'Density by layer (centre)', 'Density by layer (edge)']

    # Initialize figure
    fig = make_subplots(rows=3, cols=4, subplot_titles=titles, horizontal_spacing=0.32, vertical_spacing=0.12,
                        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                               [{"secondary_y": True}, {"secondary_y": True}, {"type": "scatter"}, {"type": "scatter"}],
                               [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]])

    # Histograms
    hist_params = ['ns', 'areas', 'meanThetas']
    max_counts = {p: 0 for p in hist_params}

    # Scatterplots (Areas vs N, Pols/dens vs N, Std pol/den vs N, theta vs N, pol vs den)
    scatter_x_params = ['ns', 'ns', 'ns', 'ns', 'medDs']
    scatter_y_params = ['areas', ['medPols', 'medDs'], ['varPols', 'varDs'], 'meanThetas', 'medPols']
    scatter_rows = [1, 2, 2, 2, 2]
    scatter_cols = [4, 1, 2, 3, 4]

    # Layer plots
    layer_params = ['p_by_layer', 'p_from_edge', 'd_by_layer', 'd_from_edge']

    # Dictionary for axis labels
    label_dict = {'ns': 'N', 'areas': 'Area (㎡)', 'medPols': 'Med. polarization', 'varPols': 'Var. polarization', 'medDs': 'Med. density (n/㎡)',
                  'varDs': 'Var. density (n/㎡)', 'meanThetas': 'Avg. θ (rad)', 'varThetas': 'Var. θ (rad)', 'p_by_layer': 'Med. polarization',
                  'p_from_edge': 'Med. polarization', 'd_by_layer': 'Med. density (n/㎡)', 'd_from_edge': 'Med. density (n/㎡)'}

    def plot_layers(csr_dict:dict[str: np.ndarray[float]], min_obs:int, max_layers:int | None, rel_idx:int):

        # Unpack vals and idcs arrays
        vals = csr_dict['data']
        idcs = csr_dict['indptr']

        # Convert csr to (n_clusters, n_layers) matrix
        layers = np.full((len(idcs) - 1, np.max(np.diff(idcs))), np.nan, dtype=np.float32)

        for i in range(len(idcs[:-1])):
            layers[i,:(idcs[i+1] - idcs[i])] = vals[idcs[i]:idcs[i+1]]

        # Find appropriate cut-off of layers using number of observations or hard cut-off
        if max_layers is None:
            # Count number of finite observations
            n_obs = np.sum(np.isfinite(layers), axis = 0)
            cutoff = np.where(n_obs < min_obs)[0][0]

        else:
            cutoff = max_layers
            
        # Generate None separated xs and ys lists
        xs = []
        ys = []
        for i in range(len(idcs[:-1])):
            n = min((idcs[i+1] - idcs[i]), cutoff)
            xs.extend(range(0, n))
            ys.extend(vals[idcs[i]:(idcs[i] + n)].tolist())

            # None separator - breaks the line between clusters
            xs.append(None)
            ys.append(None)

        del idcs, vals

        indivs_trace = go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='rgba(55,138,221,0.2)', width=0.8),
                                  showlegend=False, hoverinfo='skip', connectgaps=False, # Ensures no bridging across Nones
                                  visible=(rel_idx == 0)) 
        
        # Compute mean line
        means_x = np.arange(0, cutoff)
        means_y = np.nanmean(layers[:,:cutoff], axis = 0)

        mean_trace = go.Scatter(x=means_x, y=means_y, mode='lines', line=dict(color='#185FA5', width=2),
                                showlegend=False, connectgaps=False, visible=(rel_idx == 0))

        return indivs_trace, mean_trace

    # Iterate over frames to plot and update max_counts for histograms
    for rel_idx in tqdm(rel_frames):
        # ---FIRST ROW---

        # Histograms (N, area, thetas)
        for i, param in enumerate(hist_params):

            if param == 'meanThetas':
                bin_edges = np.linspace(min_x[param], max_x[param], n_bins + 1)
            else:
                bin_edges = np.logspace(np.log10(max(min_x[param], 1e-6)), np.log10(max_x[param]), n_bins + 1)
            counts, _ = np.histogram(data[str(rel_idx)][param], bins = bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Update max_counts
            if np.max(counts) > max_counts[param]:
                max_counts[param] = np.max(counts)

            # Add trace
            fig.add_trace(go.Bar(x=bin_centers, y=counts,                            
                                 showlegend=False,
                                 visible=(rel_idx == 0),  # only t = 0 visible initially
                                 marker_color='steelblue',
                                 marker_line_width=0), row=1, col=i + 1)
        
        # ---(END OF FIRST AND) SECOND ROW---

        for i, x_param in enumerate(scatter_x_params):
            if type(scatter_y_params[i]) == list:
                for j in range(2):
                    fig.add_trace(go.Scatter(x=data[str(rel_idx)][x_param], y=data[str(rel_idx)][scatter_y_params[i][j]], mode='markers', 
                                             marker=dict(color=['steelblue', 'coral'][j], size=5), showlegend=False, visible=(rel_idx == 0)), scatter_rows[i], scatter_cols[i], bool(j))
            else:
                fig.add_trace(go.Scatter(x=data[str(rel_idx)][x_param], y=data[str(rel_idx)][scatter_y_params[i]], mode='markers', 
                                         marker=dict(color='steelblue', size=5), showlegend=False, visible=(rel_idx == 0)), scatter_rows[i], scatter_cols[i], False)
                            
        # ---THIRD ROW---

        # Line plots (pols vs layer (center), pols vs layer (edge), dens vs layer (center), dens vs layer (edge))
        for i in range(4):
            idvs_trace, mean_trace = plot_layers(data[str(rel_idx)][layer_params[i]], min_obs, max_layers, rel_idx)
            fig.add_trace(idvs_trace, 3, i+1)
            fig.add_trace(mean_trace, 3, i+1)
        del idvs_trace, mean_trace

        data[str(rel_idx)] = None  # Reduce memory as we go
        gc.collect()

    # Build slider steps
    traces_per_frame = len(fig.data) // len(rel_frames)
    assert len(fig.data) % len(rel_frames) == 0, f"Trace count {len(fig.data)} not divisible by {len(rel_frames)} frames"

    steps = []
    for i, rel in enumerate(rel_frames):
        # Create 
        visible_mask = np.zeros(traces_per_frame * len(rel_frames)).astype(bool)

        start = i * traces_per_frame
        end = start + traces_per_frame
        visible_mask[start:end] = True

        steps.append(dict(method='restyle', args=[{'visible':visible_mask.tolist()}],
                         label=f't={rel:+d}' if rel != 0 else 't=0'))
        
    # Update figure
    fig.update_layout(sliders=[dict(active=round((len(data) - 1)/2), steps=steps,
                                   currentvalue=dict(prefix='Relative frame: ', font=dict(size=13)),
                                   pad=dict(t=40, b=10))],
                      height=900, template='plotly_white', margin=dict(l=50, r=30, t=80, b=80))

    # Histograms
    for i in range(3):
        x_scale = ['log', 'log', 'linear'][i]
        print([np.log10(max(min_x[hist_params[i]], 1e-6)), np.log10(max_x[hist_params[i]])])
        fig.update_xaxes(title_text = label_dict[hist_params[i]], range = [min_x[hist_params[i]], max_x[hist_params[i]]]
                                                                           if x_scale == 'linear'
                                                                           else [np.log10(max(min_x[hist_params[i]], 1e-6)), np.log10(max_x[hist_params[i]])], 
                         row = 1, col = i + 1, type = x_scale)
        y_scale = ['log', 'log', 'linear'][i]
        fig.update_yaxes(title_text = 'Counts', range = [0, max_counts[hist_params[i]]]
                                                         if y_scale == 'linear'
                                                         else [0.9, np.log10(max_counts[hist_params[i]])], 
                         row = 1, col = i + 1, type = y_scale)

    # Scatterplots
    for i in range(5):
        x_scale = ['log', 'log', 'log', 'log', 'linear'][i]
        fig.update_xaxes(title_text = label_dict[scatter_x_params[i]], range = [min_x[scatter_x_params[i]], max_x[scatter_x_params[i]]]
                                                                                if x_scale == 'linear'
                                                                                else [np.log10(max(min_x[scatter_x_params[i]], 1e-6)), np.log10(max_x[scatter_x_params[i]])], 
                         row = scatter_rows[i], col = scatter_cols[i], type = x_scale)

        y_scale = ['log', 'linear', 'linear', 'linear', 'linear'][i]
        if type(scatter_y_params[i]) == list:
            for j in range(2):
                fig.update_yaxes(title_text = label_dict[scatter_y_params[i][j]], range = [min_x[scatter_y_params[i][j]], max_x[scatter_y_params[i][j]]] 
                                                                                           if y_scale == 'linear' 
                                                                                           else [np.log10(max(min_x[scatter_y_params[i][j]], 1e-6)), np.log10(max_x[scatter_y_params[i][j]])], 
                                 row = scatter_rows[i], col = scatter_cols[i], secondary_y = bool(j), type = y_scale)
        else:
            fig.update_yaxes(title_text = label_dict[scatter_y_params[i]], range = [min_x[scatter_y_params[i]], max_x[scatter_y_params[i]]] 
                                                                                    if y_scale == 'linear' 
                                                                                    else [np.log10(max(min_x[scatter_y_params[i]], 1e-6)), np.log10(max_x[scatter_y_params[i]])], 
                             row = scatter_rows[i], col = scatter_cols[i], type = y_scale)
        
    # Layer plots
    for i in range(4):
        fig.update_xaxes(title_text = ['Layer (from center)', 'Layer (from edge)', 'Layer (from center)', 'Layer (from edge)'][i], row = 3, col = i + 1)
        fig.update_yaxes(title_text = label_dict[layer_params[i]], row = 3, col = i + 1, range = [min_x[layer_params[i]], max_x[layer_params[i]]])

    output_path = '.'.join(input_path.split('.')[:-1]) + '.html'
    fig.write_html(output_path)
    print(f'Saved to {output_path}')
    return fig

def interactive_cluster_merging(input_path: str):

    data = load_cluster_stats_h5(input_path)

    # Collect total clustered individuals and total number of clusters per absolute frame
    abs_frames = [int(key) for key in data.keys()]
    n_clustered = []
    n_clusters = []

    for abs in data.keys():
        ns = data[abs]['ns']
        n_clustered.append(np.sum(ns))
        n_clusters.append(len(ns))

    # Create figure
    fig = make_subplots(rows=1, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": False}]])

    # Col 1: time series
    fig.add_trace(go.Scatter(x=abs_frames, y=n_clustered, line=dict(color='steelblue'), name="Clustered individuals"), secondary_y=False, row=1, col=1)
    fig.add_trace(go.Scatter(x=abs_frames, y=n_clusters, line=dict(color='coral'), name="Clusters"), secondary_y=True, row=1, col=1)

    # Col 2: n_clustered vs n_clusters scatter
    fig.add_trace(go.Scatter(x=n_clustered, y=n_clusters, mode='markers', marker=dict(color=abs_frames, colorscale='Viridis', colorbar=dict(title='Absolute frame', orientation='h', x=0.72, y=-0.1, xanchor='center', yanchor='top',
                                                                                                                                            len=0.45, thickness=20), showscale=True), name="Single frame"), secondary_y=False, row=1, col=2)

    # Add range slider to col 1 x-axis only
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True), type='linear'),
                    yaxis=dict(anchor="x", autorange=True, mirror=True, showline=True, side="left", tickmode="auto", ticks="", type="linear", zeroline=False),
                    yaxis2=dict(anchor="x", autorange=True, mirror=True, showline=True, side="right", tickmode="auto", ticks="", type="linear", zeroline=False))

    # Update labels — use col to target the right axis
    fig.update_xaxes(title_text='Frame', row=1, col=1)
    fig.update_xaxes(title_text='Total number of clustered individuals', row=1, col=2)
    fig.update_yaxes(title_text='Total number of clustered individuals', secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text='Total number of clusters', secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text='Total number of clusters', row=1, col=2)

    output_path = '.'.join(input_path.split('.')[:-1]) + '_merging.html'
    fig.write_html(output_path)
    print(f'Saved to {output_path}')
    return fig

def interactive_cluster_structure(ds:xr.Dataset, input_path:str, layer_cutoff:int| None = None, fps:int = 5, start_frame:int = 0, end_frame:int | None = None, subsample:int = 1):

    # Load data
    data = load_cluster_stats_h5(input_path)

    # Define stat names that will be aggregated and initialize storage dictionary
    stat_names = ['p_by_layer', 'p_from_edge', 'd_by_layer', 'd_from_edge']

    # Store data according to max_layer value and relative frame - first by max_layer, then by stat name
    stats_by_max_layer:dict[int, dict[str, list]] = {}

    # Iterate over absolute frames
    for j, key in enumerate(data.keys()):

        # Iterate over stat type
        for stat in stat_names:

            # Get observations for this absolute frame
            csr_dict = data[key][stat]

            # Unpack vals and idcs arrays
            vals = csr_dict['data']
            idcs = csr_dict['indptr']

            # Add lists to dictionary
            for i in range(len(idcs) - 1):
                max_layer = idcs[i+1] - idcs[i]
                stats_by_max_layer.setdefault(int(max_layer), {}).setdefault(stat, []).append(vals[idcs[i]:idcs[i+1]])

    def plot_layers(stats_by_max_layer:dict[int, dict[str, list]], max_layer:int, stat:str, cutoff:int | None = None):

        # Generate None separated xs and ys lists
        xs = []
        ys = []
        for row in stats_by_max_layer[max_layer][stat]:
            y = row[:cutoff]
            xs.extend(range(len(y)))
            ys.extend(y)

            # None separator - breaks the line between clusters
            xs.append(None)
            ys.append(None)

        # Create plotly trace for individual cluster curves
        indivs_trace = go.Scatter(x=xs, y=ys, mode='lines', line=dict(color="#FFF0B8", width=0.8), showlegend=False, hoverinfo='skip', connectgaps=False, visible=(max_layer == 5))

        # Take median of all cluster curves as a function of layer
        
        
        median_y = np.nanmedian(stats_by_max_layer[max_layer][stat], axis = 0)[:cutoff]
        median_x = np.arange(len(median_y))

        # Create plotly trace for median curve
        median_trace = go.Scatter(x=median_x, y=median_y, mode='lines', line=dict(color="#1172D2", width=2), showlegend=False, connectgaps=False, visible=(max_layer == 5))

        return indivs_trace, median_trace, np.nanmax(stats_by_max_layer[max_layer][stat])
    
    # Initialize figure and variables to store extrema
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.16, vertical_spacing=0.12)
    max_x = layer_cutoff if layer_cutoff else np.max(list(stats_by_max_layer.keys()))
    max_y = {stat: [1, 1, 0, 0] for stat in stat_names}
    all_max_ds = []
    
    # Iterate over unique n values and plot (including adding slider steps)
    unique_max_layers = np.unique(list(stats_by_max_layer.keys()))
    steps = []
    traces_per_frame = 2*len(stat_names) # 2: Individuals, median

    for j, max_layer in enumerate(unique_max_layers):

        # Iterate over different stats
        for i, stat in enumerate(stat_names):

            # Get traces
            indivs, median, maxy = plot_layers(stats_by_max_layer, max_layer, stat, layer_cutoff)

            # Update extrema
            if stat == 'd_by_layer': # Don't need to do it for d_from_edge, since values are the same
                all_max_ds.append(maxy)

            # Add traces to figure
            fig.add_trace(indivs, row= (i // 2) + 1, col= (i % 2) +1)
            fig.add_trace(median, row= (i // 2) + 1, col= (i % 2) +1)
    
        # Create a visibility mask
        visible_mask = np.zeros(traces_per_frame * len(unique_max_layers)).astype(bool)
        start = j*traces_per_frame
        end = start + traces_per_frame
        visible_mask[start:end] = True

        steps.append(dict(method='restyle', args=[{'visible':visible_mask.tolist()}], label=f'l = {max_layer}'))

    # Update maximum density values using quantiles (to exclude outliers)
    for stat in ['d_by_layer', 'd_from_edge']:
        max_y[stat] = np.quantile(all_max_ds, 0.9)

    # Update figure
    fig.update_layout(sliders=[dict(active= 5, steps=steps, currentvalue=dict(prefix='Layer radius: ', font=dict(size=13)), pad=dict(t=40, b=10))], height=900, template='plotly_white', margin=dict(l=50, r=30, t=80, b=80))
    
    
    # Update axes of figure
    row_labels = ['Median polarization', 'Median density']
    col_labels = ['Voronoi layer (from center)', 'Voronoi layer (from edge)']
    for i in range(4):
        fig.update_xaxes(title_text=col_labels[i%2] if i > 1 else None, range=[0, max_x], row=(i//2)+1, col=(i%2)+1)
        fig.update_yaxes(title_text=row_labels[i//2] if not (i%2) else None, range=[0, max_y[stat_names[i]]], row=(i//2)+1, col=(i%2)+1)

    # Save figure
    output_path = '.'.join(input_path.split('.')[:-1]) + '_structure.html'
    fig.write_html(output_path)
    print(f'Saved to {output_path}')
    return fig

def plot_distribution_over_time_interactive(ds: xr.Dataset, y_var: str, y_label: str, output_dir: str, title: str, batch_num:int, y_factor: float = 1, fps: int = 5, start_frame: int = 0, end_frame: int = None, y_bins: int = 50, y_quant: float = 1, 
                                            time_bins: int = 200, subsample: int = 1):
    """Plot distribution over time with an interactive PDF for each time bin."""

    # Get absolute frame values
    abs_frames, ds_idcs = get_frame_slice(ds, rel_start=start_frame, rel_end=end_frame, in_function_subsample=subsample)

    dist_array = ds[y_var].values[ds_idcs, :] * y_factor
    _, n_ids = dist_array.shape

    # Coordinates for every data point
    x = np.repeat(abs_frames, n_ids) / fps
    y = dist_array.flatten()

    # Remove NaNs and values above requested quantile
    y_max = np.nanquantile(y, y_quant)
    mask = (~np.isnan(y)) & (y <= y_max)

    x_clean = x[mask]
    y_clean = y[mask]

    # Compute histogram explicitly so both panels use identical bins
    H, x_edges, y_edges = np.histogram2d(x_clean, y_clean, bins=[time_bins, y_bins])

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    y_widths = np.diff(y_edges)

    # Convert counts to log10 for heatmap visualization
    # NaNs make zero-count bins transparent
    H_log = np.where(H.T > 0, np.log10(H.T), np.nan)

    # Initial PDF
    counts = H[0]

    if counts.sum() > 0:
        pdf = counts / (counts.sum() * y_widths)
    else:
        pdf = np.zeros_like(counts)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.72, 0.28], horizontal_spacing=0.20, subplot_titles=("Distribution over time", "Distribution at selected time"))

    # Left: 2D histogram
    fig.add_trace(go.Heatmap(x=x_centers, y=y_centers, z=H_log, colorscale="Magma", colorbar=dict(title="log10(Num. individuals)", x=0.59),
                  hovertemplate=("Time: %{x:.2f} s<br>" + y_label + ": %{y:.3g}<br>" "log10(count): %{z:.2f}" "<extra></extra>"),), row=1, col=1)

    # Right: PDF, horizontal so it shares the y variable
    fig.add_trace(go.Scatter(x=pdf, y=y_centers, mode="lines", fill="tozerox", name="PDF", hovertemplate=(y_label + ": %{y:.3g}<br>" "PDF: %{x:.3g}" "<extra></extra>")), row=1, col=2)

    # Frames update the PDF and vertical line
    frames = []

    for i, time in enumerate(x_centers):

        counts = H[i]

        if counts.sum() > 0:
            pdf_i = counts / (counts.sum() * y_widths)
        else:
            pdf_i = np.zeros_like(counts)

        frames.append(go.Frame(name=str(i), data=[go.Scatter(x=pdf_i, y=y_centers)], traces=[1], layout=go.Layout(shapes=[dict(type="line", x0=time, x1=time, y0=0, y1=1, xref="x", yref="paper",
                                                                                                                               line=dict(width=3, color="white"))])))

    fig.frames = frames

    # Slider
    slider_steps = []

    for i, time in enumerate(x_centers):

        # Approximate absolute frame represented by this time bin
        frame = int(round(time * fps))

        slider_steps.append(dict(method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))], label=str(frame)))

    sliders = [
        dict(
            active=0,
            currentvalue=dict(
                prefix="Frame: ",
                font=dict(size=14),
            ),
            pad=dict(t=50),
            steps=slider_steps,
        )
    ]

    # Initial vertical line
    initial_shape = dict(
        type="line",
        x0=x_centers[0],
        x1=x_centers[0],
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            width=3,
            color="white",
        ),
    )

    fig.update_layout(
        title=title,
        sliders=sliders,
        shapes=[initial_shape],
        height=650,
        width=1300,
        template="plotly_white",
        showlegend=False,
    )

    fig.update_xaxes(
        title_text="Experiment time (s)",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text=y_label,
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Probability density",
        row=1,
        col=2,
    )

    # Match y ranges exactly between panels
    fig.update_yaxes(
        range=[y_edges[0], y_edges[-1]],
        row=1,
        col=1,
    )

    fig.update_yaxes(
        range=[y_edges[0], y_edges[-1]],
        row=1,
        col=2,
    )

    # Save as interactive HTML
    save_dir = os.path.join(output_dir, f"hists_over_time/sliders/batch_{batch_num}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"{y_var}_{abs_frames[0]}_{abs_frames[-1]}_interactive.html",
    )

    fig.write_html(save_path)

    print(f"Interactive histogram saved to {save_path}")

    return fig


def plot_distribution_over_position_interactive(
    ds: xr.Dataset,
    z_var: str,
    output_dir: str,
    batch_num: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    z_label: str | None = None,
    title: str | None = None,
    
    y_factor: float = 1,
    subsample: int = 1,
    x_bins: int = 50,
    y_bins: int = 50,
    value_bins: int = 50,
    value_quantile: float = 1):
    """Plot the distribution of a variable over x and y position with a frame slider."""

    if z_label is None:
        z_label = z_var

    if title is None:
        title = f"Distribution of {z_var} over position"

    # Get absolute frame values and corresponding dataset indices
    abs_frames, ds_idcs = get_frame_slice(
        ds,
        rel_start=start_frame,
        rel_end=end_frame,
        in_function_subsample=subsample,
    )

    values = ds[z_var].values[ds_idcs, :] * y_factor
    x_positions = ds['centroid_x'].values[ds_idcs, :]
    y_positions = ds['centroid_y'].values[ds_idcs, :]

    # Determine common value limits and bin edges
    value_min = np.nanmin(values)
    value_max = np.nanquantile(values, value_quantile)

    value_edges = np.linspace(value_min, value_max, value_bins + 1)
    x_edges = np.linspace(np.nanmin(x_positions), np.nanmax(x_positions), x_bins + 1)
    y_edges = np.linspace(np.nanmin(y_positions), np.nanmax(y_positions), y_bins + 1)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    value_centers = (value_edges[:-1] + value_edges[1:]) / 2

    # Compute one histogram for each frame and position coordinate
    x_histograms = []
    y_histograms = []

    for frame_values, frame_x, frame_y in zip(values, x_positions, y_positions):
        valid_x = (
            np.isfinite(frame_values)
            & np.isfinite(frame_x)
            & (frame_values <= value_max)
        )
        valid_y = (
            np.isfinite(frame_values)
            & np.isfinite(frame_y)
            & (frame_values <= value_max)
        )

        H_x, _, _ = np.histogram2d(
            frame_x[valid_x],
            frame_values[valid_x],
            bins=[x_edges, value_edges],
        )
        H_y, _, _ = np.histogram2d(
            frame_y[valid_y],
            frame_values[valid_y],
            bins=[y_edges, value_edges],
        )

        x_histograms.append(H_x.T)
        y_histograms.append(H_y.T)

    x_histograms = np.asarray(x_histograms)
    y_histograms = np.asarray(y_histograms)

    # Use one common colour scale for both subplots and all frames
    max_count = max(
        np.nanmax(x_histograms),
        np.nanmax(y_histograms),
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        subplot_titles=(
            f"{z_label} distribution over {'x (m)'}",
            f"{z_label} distribution over {'y (m)'}",
        ),
    )

    # Initial heatmaps
    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=value_centers,
            z=x_histograms[0],
            zmin=0,
            zmax=max_count,
            colorscale="Magma",
            colorbar=dict(title="Count", x=0.46),
            hovertemplate=(
                f"{'x (m)'}: %{{x:.3g}}<br>"
                f"{z_label}: %{{y:.3g}}<br>"
                "Count: %{z}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=y_centers,
            y=value_centers,
            z=y_histograms[0],
            zmin=0,
            zmax=max_count,
            colorscale="Magma",
            showscale=False,
            hovertemplate=(
                f"{'y (m)'}: %{{x:.3g}}<br>"
                f"{z_label}: %{{y:.3g}}<br>"
                "Count: %{z}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    # Create slider frames
    frames = []

    for i, frame in enumerate(abs_frames):
        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Heatmap(z=x_histograms[i]),
                    go.Heatmap(z=y_histograms[i]),
                ],
                traces=[0, 1],
            )
        )

    fig.frames = frames

    # Create slider
    slider_steps = []

    for i, frame in enumerate(abs_frames):
        slider_steps.append(
            dict(
                method="animate",
                args=[
                    [str(i)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=str(frame),
            )
        )

    fig.update_layout(
        title=title,
        height=650,
        width=1300,
        template="plotly_white",
        showlegend=False,
        sliders=[
            dict(
                active=0,
                currentvalue=dict(
                    prefix="Frame: ",
                    font=dict(size=14),
                ),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
    )

    fig.update_xaxes(
        title_text='x (m)',
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text='y (m)',
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text=z_label,
        range=[value_edges[0], value_edges[-1]],
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text=z_label,
        range=[value_edges[0], value_edges[-1]],
        row=1,
        col=2,
    )

    # Save as interactive HTML
    if output_dir is not None:
        if batch_num is None:
            save_dir = os.path.join(
                output_dir,
                "hists_over_axis",
            )
        else:
            save_dir = os.path.join(
                output_dir,
                f"hists_over_axis/sliders/batch_{batch_num}",
            )

        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            f"{z_var}_{abs_frames[0]}_{abs_frames[-1]}_interactive.html",
        )

        fig.write_html(save_path)
        print(f"Interactive histogram saved to {save_path}")

    return fig

def plot_position_histograms_interactive(
    ds: xr.Dataset,
    output_dir: str,
    batch_num: int,
    title: str = "Locust positions over time",
    start_frame: int = 0,
    end_frame: int | None = None,
    x_bins: int = 50,
    y_bins: int = 50,
    subsample: int = 1,
):
    """Plot counts over binned x and y positions with a frame slider."""

    abs_frames, ds_idcs = get_frame_slice(
        ds,
        rel_start=start_frame,
        rel_end=end_frame,
        in_function_subsample=subsample,
    )

    x_positions = ds["centroid_x"].values[ds_idcs, :]
    y_positions = ds["centroid_y"].values[ds_idcs, :]

    x_edges = np.linspace(np.nanmin(x_positions), np.nanmax(x_positions), x_bins + 1)
    y_edges = np.linspace(np.nanmin(y_positions), np.nanmax(y_positions), y_bins + 1)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    x_counts = []
    y_counts = []

    for frame_x, frame_y in zip(x_positions, y_positions):
        x_counts.append(np.histogram(frame_x[np.isfinite(frame_x)], bins=x_edges)[0])
        y_counts.append(np.histogram(frame_y[np.isfinite(frame_y)], bins=y_edges)[0])

    x_counts = np.asarray(x_counts)
    y_counts = np.asarray(y_counts)
    max_count = max(x_counts.max(), y_counts.max())

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
        subplot_titles=("Distribution over x position", "Distribution over y position"),
    )

    fig.add_trace(
        go.Bar(
            x=x_centers,
            y=x_counts[0],
            width=np.diff(x_edges),
            marker_color="rgba(128, 0, 0, 0.8)",
            hovertemplate="x: %{x:.3g}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=y_centers,
            y=y_counts[0],
            width=np.diff(y_edges),
            marker_color="rgba(128, 0, 0, 0.8)",
            hovertemplate="y: %{x:.3g}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Update both histograms for each selected frame
    fig.frames = [
        go.Frame(
            name=str(i),
            data=[
                go.Bar(y=x_counts[i]),
                go.Bar(y=y_counts[i]),
            ],
            traces=[0, 1],
        )
        for i in range(len(abs_frames))
    ]

    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=str(frame),
        )
        for i, frame in enumerate(abs_frames)
    ]

    fig.update_layout(
        title=title,
        height=650,
        width=1300,
        template="plotly_white",
        showlegend=False,
        barmode="overlay",
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Frame: ", font=dict(size=14)),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
    )

    fig.update_xaxes(title_text="x (m)", row=1, col=1)
    fig.update_xaxes(title_text="y (m)", row=1, col=2)
    fig.update_yaxes(title_text="Number of individuals", range=[0, max_count * 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Number of individuals", range=[0, max_count * 1.05], row=1, col=2)

    if output_dir is not None:
        save_dir = os.path.join(
            output_dir,
            "position_histograms/sliders"
            if batch_num is None
            else f"position_histograms/sliders/batch_{batch_num}",
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(
            save_dir,
            f"xy_hists_{abs_frames[0]}_{abs_frames[-1]}_interactive.html",
        )
        fig.write_html(save_path)
        print(f"Interactive position histograms saved to {save_path}")

    return fig

def plot_reflection_aligned_position_distributions(
    ds: xr.Dataset,
    output_dir: str,
    batch_num: int,
    arena_center_m: np.ndarray = np.array([0.01, 0]),
    fps: float = 5,
    title: str = "Position distributions around reflection events",
    start_frame: int = 0,
    end_frame: int | None = None,
    x_bins: int = 50,
    y_bins: int = 50,
    subsample: int = 1
):
    """Plot median x/y position distributions around left and right reflections."""

    # Load all necessary data from dataset
    abs_frames, ds_idcs = get_frame_slice(ds, rel_start=start_frame, rel_end=end_frame, in_function_subsample=subsample)
    x_positions = ds['centroid_x'].values[ds_idcs, :]
    y_positions = ds['centroid_y'].values[ds_idcs, :]

    # Find reflections
    rel_reflections, _, marching_period, reflection_sides = find_reflections(ds, fps, start_frame, end_frame, arena_center_m, subsample)
    print(f'Marching period (frames): {marching_period:.2f}')
    print(f'Marching period (s): {(marching_period/fps):.2f}')

    # Determine if dataset is subsampled already and define relative frames to events
    frame_step = max(1, int(round(np.median(np.diff(abs_frames)))))
    half_period = marching_period // 2
    relative_frames = np.arange(-half_period, half_period + 1, frame_step)

    # Keep only events with a complete relative-frame window
    usable_events = [i for i, event_frame in enumerate(rel_reflections) if all(int(event_frame + relative_frame) in ds_idcs for relative_frame in relative_frames)]

    # Remove the final event if needed to obtain equal left/right event counts
    if len(usable_events) % 2 != 0:
        usable_events = usable_events[:-1]

    left_events = [i for i in usable_events if reflection_sides[i] == "left"]
    right_events = [i for i in usable_events if reflection_sides[i] == "right"]
    print(left_events)
    print(right_events)

    if len(left_events) != len(right_events) or len(left_events) == 0:
        raise ValueError("Could not obtain equal non-zero numbers of left and right reflections.")

    # Define x bins symmetrically around the arena centre
    x_limit = max(
        abs(np.nanmin(x_positions)),
        abs(np.nanmax(x_positions)),
    )
    x_edges = np.linspace(-x_limit, x_limit, x_bins + 1)
    y_edges = np.linspace(
        np.nanmin(y_positions),
        np.nanmax(y_positions),
        y_bins + 1,
    )

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2


    def get_summary(position_data, event_indices, bin_edges, reflect=False):
        medians, lower, upper = [], [], []

        for relative_frame in relative_frames:
            event_counts = []

            for event_index in event_indices:
                event_frame = rel_reflections[event_index]
                frame_index = int(event_frame + relative_frame)

                positions = position_data[frame_index]

                if reflect:
                    positions = -positions

                positions = positions[np.isfinite(positions)]
                counts = np.histogram(positions, bins=bin_edges)[0]
                event_counts.append(counts)

            event_counts = np.asarray(event_counts)
            medians.append(np.median(event_counts, axis=0))
            lower.append(np.percentile(event_counts, 25, axis=0))
            upper.append(np.percentile(event_counts, 75, axis=0))

        return np.asarray(medians), np.asarray(lower), np.asarray(upper)

    # Mirror left-reflection x positions into the right-reflection coordinate system
    x_left = get_summary(x_positions, left_events, x_edges, reflect=True)
    x_right = get_summary(x_positions, right_events, x_edges, reflect=False)
    y_left = get_summary(y_positions, left_events, y_edges, reflect=False)
    y_right = get_summary(y_positions, right_events, y_edges, reflect=False)

    summaries = [x_left, x_right, y_left, y_right]
    colours = ["royalblue", "firebrick", "royalblue", "firebrick"]

    def make_trace(summary, position_values, colour, name, showlegend):
        medians, lower, upper = summary

        return go.Scatter(
            x=position_values,
            y=medians[0],
            mode="lines+markers",
            line=dict(color=colour, width=2),
            marker=dict(size=5),
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            error_y=dict(
                type="data",
                symmetric=False,
                array=upper[0] - medians[0],
                arrayminus=medians[0] - lower[0],
                thickness=1,
                width=3,
            ),
            hovertemplate=(
                "Position: %{x:.3g}<br>"
                "Median count: %{y:.3g}"
                "<extra></extra>"
            ),
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(
            "Reflection-aligned x-position",
            "Reflection-aligned y-position",
        ),
    )
    names = ['Left reflection', 'Right reflection']
    fig.add_trace(
        make_trace(x_left, x_centers, colours[0], names[0], True),
        row=1,
        col=1,
    )
    fig.add_trace(
        make_trace(x_right, x_centers, colours[1], names[1], True),
        row=1,
        col=1,
    )
    fig.add_trace(
        make_trace(y_left, y_centers, colours[0], names[0], False),
        row=1,
        col=2,
    )
    fig.add_trace(
        make_trace(y_right, y_centers, colours[1], names[1], False),
        row=1,
        col=2,
    )

    # Update all four traces for each relative frame
    fig.frames = [
        go.Frame(
            name=str(relative_frame),
            data=[
                go.Scatter(
                    y=summary[0][i],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=summary[2][i] - summary[0][i],
                        arrayminus=summary[0][i] - summary[1][i],
                        thickness=1,
                        width=3,
                    ),
                )
                for summary in summaries
            ],
            traces=[0, 1, 2, 3],
        )
        for i, relative_frame in enumerate(relative_frames)
    ]

    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(relative_frame)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=str(relative_frame),
        )
        for relative_frame in relative_frames
    ]

    x_y_max = max(
        np.max(x_left[2]),
        np.max(x_right[2]),
    )
    y_y_max = max(
        np.max(y_left[2]),
        np.max(y_right[2]),
    )

    fig.update_layout(
        title=f"{title} (estimated period: {marching_period} frames)",
        height=650,
        width=1300,
        template="plotly_white",
        hovermode="closest",
        sliders=[
            dict(
                active=0,
                currentvalue=dict(
                    prefix="Relative frame: ",
                    font=dict(size=14),
                ),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
    )

    fig.update_xaxes(
        title_text="Position relative to arena centre",
        range=[-x_limit, x_limit],
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="y position",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text="Median locust count",
        range=[0, x_y_max * 1.05],
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Median locust count",
        range=[0, y_y_max * 1.05],
        row=1,
        col=2,
    )

    if output_dir is not None:
        save_dir = os.path.join(
            output_dir,
            "reflection_aligned_distributions"
            if batch_num is None
            else f"reflection_aligned_distributions/batch_{batch_num}",
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"reflection_aligned_position_distributions_{abs_frames[0]}_{abs_frames[-1]}.html")
        fig.write_html(save_path)
        print(f"Reflection-aligned distributions saved to {save_path}")

    print(f"Included reflection events: {len(left_events)} left and {len(right_events)} right")

    return fig

def plot_reflection_aligned_variable_distributions(
    ds: xr.Dataset,
    variable: str,
    variable_name: str,
    x_var: str,
    y_var: str,
    output_dir: str,
    batch_num: int,
    arena_center_m: np.ndarray = np.array([0.01, 0]),
    fps: float = 5,
    title: str | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    x_bins: int = 40,
    y_bins: int = 40,
    value_bins: int = 40,
    value_range: tuple[float, float] | None = None,
    subsample: int = 1,
    period:int = 465,
):
    """Plot probability distributions of a variable over x and y around reflections."""
    

    # Load all necessary data from dataset
    abs_frames, ds_idcs = get_frame_slice(ds, rel_start=start_frame, rel_end=end_frame, in_function_subsample=subsample)
    x_positions = ds[x_var].values[ds_idcs, :]
    y_positions = ds[y_var].values[ds_idcs, :]
    variable_values = ds[variable].values[ds_idcs, :]

    # Find reflections
    rel_reflections, _, marching_period, reflection_sides = find_reflections(ds, fps, start_frame, end_frame, arena_center_m, subsample)
    print(f'Marching period (frames): {marching_period:.2f}')
    print(f'Marching period (s): {(marching_period/fps):.2f}')

    frame_step = max(1, int(round(np.median(np.diff(abs_frames)))))
    half_period = marching_period // 2
    relative_frames = np.arange(-half_period, half_period + 1, frame_step)

    # Keep only events with a complete relative-frame window
    usable_events = [i for i, event_frame in enumerate(rel_reflections) if all(int(event_frame + relative_frame) in ds_idcs for relative_frame in relative_frames)]

    # Remove the final event if needed to obtain equal left/right event counts
    if len(usable_events) % 2 != 0:
        usable_events = usable_events[:-1]

    left_events = [i for i in usable_events if reflection_sides[i] == "left"]
    right_events = [i for i in usable_events if reflection_sides[i] == "right"]

    # Truncate both sides to the same number of events
    n_events = min(len(left_events), len(right_events))
    left_events = left_events[:n_events]
    right_events = right_events[:n_events]

    if n_events == 0:
        raise ValueError("Could not obtain equal non-zero numbers of reflections.")

    # Position bins remain in the physical coordinate system; x is not reflected
    x_finite = x_positions[np.isfinite(x_positions)]
    y_finite = y_positions[np.isfinite(y_positions)]

    x_edges = np.linspace(np.nanmin(x_finite), np.nanmax(x_finite), x_bins + 1)
    y_edges = np.linspace(np.nanmin(y_finite), np.nanmax(y_finite), y_bins + 1)

    # Infer the variable range unless it is supplied explicitly
    variable_finite = variable_values[np.isfinite(variable_values)]

    if value_range is None:
        value_min = np.nanmin(variable_finite)
        value_max = np.nanmax(variable_finite)
    else:
        value_min, value_max = value_range

    if value_min >= value_max:
        raise ValueError("value_range must contain two increasing values.")

    value_edges = np.linspace(value_min, value_max, value_bins + 1)

    def get_probability_histograms(event_indices):
        """Pool observations across events and normalize each frame to probability."""
        histograms = []

        for relative_frame in relative_frames:
            all_positions_x = []
            all_positions_y = []
            all_values_x = []
            all_values_y = []

            for event_index in event_indices:
                event_frame = rel_reflections[event_index]
                frame_index = int(event_frame + relative_frame)

                frame_x = x_positions[frame_index]
                frame_y = y_positions[frame_index]
                frame_values = variable_values[frame_index]

                valid_x = (
                    np.isfinite(frame_x)
                    & np.isfinite(frame_values)
                    & (frame_values >= value_min)
                    & (frame_values <= value_max)
                )
                valid_y = (
                    np.isfinite(frame_y)
                    & np.isfinite(frame_values)
                    & (frame_values >= value_min)
                    & (frame_values <= value_max)
                )

                all_positions_x.append(frame_x[valid_x])
                all_values_x.append(frame_values[valid_x])
                all_positions_y.append(frame_y[valid_y])
                all_values_y.append(frame_values[valid_y])

            positions_x = np.concatenate(all_positions_x)
            values_x = np.concatenate(all_values_x)
            positions_y = np.concatenate(all_positions_y)
            values_y = np.concatenate(all_values_y)

            hist_x, _, _ = np.histogram2d(
                positions_x,
                values_x,
                bins=[x_edges, value_edges],
            )
            hist_y, _, _ = np.histogram2d(
                positions_y,
                values_y,
                bins=[y_edges, value_edges],
            )

            hist_x = hist_x.T
            hist_y = hist_y.T

            if hist_x.sum() > 0:
                hist_x = hist_x / hist_x.sum()

            if hist_y.sum() > 0:
                hist_y = hist_y / hist_y.sum()

            histograms.append((hist_x, hist_y))

        return histograms

    left_histograms = get_probability_histograms(left_events)
    right_histograms = get_probability_histograms(right_events)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    value_centers = (value_edges[:-1] + value_edges[1:]) / 2

    all_histograms = [
        histogram
        for event_histograms in [left_histograms, right_histograms]
        for histogram_pair in event_histograms
        for histogram in histogram_pair
    ]
    max_probability = max(np.nanmax(histogram) for histogram in all_histograms)

    if title is None:
        title = f"{variable_name} distributions around reflections"

    fig = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
        subplot_titles=(
            f"Left reflections: {variable_name} over x",
            f"Left reflections: {variable_name} over y",
            f"Right reflections: {variable_name} over x",
            f"Right reflections: {variable_name} over y",
        ),
    )

    def make_heatmap(z, x_values, show_colorbar):
        return go.Heatmap(
            x=x_values,
            y=value_centers,
            z=z,
            zmin=0,
            zmax=max_probability,
            colorscale="Magma",
            showscale=show_colorbar,
            colorbar=dict(title="Probability") if show_colorbar else None,
            hovertemplate=(
                "Position: %{x:.3g}<br>"
                f"{variable_name}: %{{y:.3g}}<br>"
                "Probability: %{z:.3g}<extra></extra>"
            ),
        )

    initial_left_x, initial_left_y = left_histograms[0]
    initial_right_x, initial_right_y = right_histograms[0]

    fig.add_trace(make_heatmap(initial_left_x, x_centers, True), row=1, col=1)
    fig.add_trace(make_heatmap(initial_left_y, y_centers, False), row=1, col=2)
    fig.add_trace(make_heatmap(initial_right_x, x_centers, False), row=2, col=1)
    fig.add_trace(make_heatmap(initial_right_y, y_centers, False), row=2, col=2)

    # Update all four heatmaps for each relative frame
    fig.frames = [
        go.Frame(
            name=str(relative_frame),
            data=[
                go.Heatmap(z=left_histograms[i][0]),
                go.Heatmap(z=left_histograms[i][1]),
                go.Heatmap(z=right_histograms[i][0]),
                go.Heatmap(z=right_histograms[i][1]),
            ],
            traces=[0, 1, 2, 3],
        )
        for i, relative_frame in enumerate(relative_frames)
    ]

    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(relative_frame)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=str(relative_frame),
        )
        for relative_frame in relative_frames
    ]

    fig.update_layout(
        title=f"{title} (estimated period: {marching_period} frames)",
        height=900,
        width=1300,
        template="plotly_white",
        showlegend=False,
        sliders=[
            dict(
                active=0,
                currentvalue=dict(
                    prefix="Relative frame: ",
                    font=dict(size=14),
                ),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
    )

    fig.update_xaxes(title_text="x position", row=1, col=1)
    fig.update_xaxes(title_text="y position", row=1, col=2)
    fig.update_xaxes(title_text="x position", row=2, col=1)
    fig.update_xaxes(title_text="y position", row=2, col=2)

    fig.update_yaxes(title_text=variable_name, range=[value_min, value_max], row=1, col=1)
    fig.update_yaxes(title_text=variable_name, range=[value_min, value_max], row=1, col=2)
    fig.update_yaxes(title_text=variable_name, range=[value_min, value_max], row=2, col=1)
    fig.update_yaxes(title_text=variable_name, range=[value_min, value_max], row=2, col=2)

    save_dir = os.path.join(output_dir, f"reflection_aligned_variable_distributions/batch_{batch_num}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{variable}_reflection_aligned_distributions.html")
    fig.write_html(save_path)
    print(f"Reflection-aligned variable distributions saved to {save_path}")

    print(f"Estimated reflection period: {marching_period} frames")
    print(f"Included events: {n_events} left and {n_events} right")

    return fig