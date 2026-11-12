"""Dash viewer for synchronized video frames and locust trajectories."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, Input, Output, dcc, html
from flask import abort, send_file

from data_handling import load_preprocessed_data

def locust_trajectory_viewer(
    image_dir,
    ds,
    *,
    x_var="x",
    y_var="y",
    z_var="z",
    history=10,
    line_alpha=0.35,
    scatter_alpha=0.5,
    image_extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    width=1200,
    height=700,
):
    """Create a Dash app with an image and a 3D plot for each selected frame.

    Images are sorted by filename and paired by position with ``ds.frame``.
    If there are more images than dataset frames, ``ds.frame`` values are used
    as zero-based indices into the sorted image list.
    """

    image_dir = Path(image_dir)
    extensions = {ext.lower() for ext in image_extensions}
    all_image_paths = sorted(
        (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: p.name,
    )
    if not all_image_paths:
        raise ValueError(f"No supported images found in {image_dir}")

    frame_values = np.asarray(ds.coords["frame"].values)
    ids = np.asarray(ds.coords["id"].values)
    if len(all_image_paths) == len(frame_values):
        image_paths = all_image_paths
    else:
        frame_indices = np.asarray(frame_values, dtype=np.int64)
        if (
            np.any(frame_indices < 0)
            or np.any(frame_indices >= len(all_image_paths))
            or not np.array_equal(frame_indices, frame_indices.astype(int))
        ):
            raise ValueError(
                f"Found {len(all_image_paths)} images and {len(frame_values)} dataset frames. "
                "When counts differ, ds.frame must contain valid zero-based image indices."
            )
        image_paths = [all_image_paths[int(i)] for i in frame_indices]

    coords = {}
    axis_ranges = {}
    for name in (x_var, y_var, z_var):
        values = np.asarray(ds[name].transpose("frame", "id").values, dtype=float)
        if values.shape != (len(frame_values), len(ids)):
            raise ValueError(f"{name!r} must have dimensions ('frame', 'id')")
        coords[name] = values

        finite = values[np.isfinite(values)]
        if finite.size == 0:
            axis_ranges[name] = (-1.0, 1.0)
            continue
        low = float(np.min(finite))
        high = float(np.max(finite))
        if high > np.nanmedian(finite) + 5 * float(np.std(finite)):
            high = float(np.quantile(finite, 0.95))
        if low < np.nanmedian(finite) - 5*np.std(finite):
            low = float(np.quantile(finite, 0.05))
        pad = max(abs(low) * 0.05, 0.5) if low == high else (high - low) * 0.04
        axis_ranges[name] = (low - pad, high + pad)

    history = int(history)
    if history < 0:
        raise ValueError("history must be non-negative")

    def make_figure(frame_index):
        start = max(0, frame_index - history)
        # Pack all locust paths into one trace, separated by NaNs, to avoid
        # creating thousands of individual WebGL traces.
        path_values = {}
        for name in (x_var, y_var, z_var):
            window = coords[name][start : frame_index + 1, :].T
            separated = np.full((window.shape[0], window.shape[1] + 1), np.nan)
            separated[:, :-1] = window
            path_values[name] = separated.ravel()[:-1]

        current = {
            name: coords[name][frame_index, :]
            for name in (x_var, y_var, z_var)
        }
        valid = np.logical_and.reduce(
            [np.isfinite(current[name]) for name in (x_var, y_var, z_var)]
        )
        id_labels = np.asarray([str(value) for value in ids], dtype=object)

        return go.Figure(
            data=[
                go.Scatter3d(
                    x=path_values[x_var],
                    y=path_values[y_var],
                    z=path_values[z_var],
                    mode="lines",
                    line={"width": 3, "color": f"rgba(42,111,187,{line_alpha})"},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                go.Scatter3d(
                    x=current[x_var][valid],
                    y=current[y_var][valid],
                    z=current[z_var][valid],
                    customdata=id_labels[valid],
                    mode="markers",
                    marker={"size": 4, "color": "#e4572e", "opacity": scatter_alpha},
                    hovertemplate="id=%{customdata}<extra></extra>",
                    showlegend=False,
                ),
            ],
            layout={
                "width": width,
                "height": height,
                "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
                "title": f"Frame {frame_values[frame_index]}",
                "scene": {
                    "xaxis": {"title": x_var, "range": axis_ranges[x_var]},
                    "yaxis": {"title": y_var, "range": axis_ranges[y_var]},
                    "zaxis": {"title": z_var, "range": axis_ranges[z_var]},
                    "uirevision": "fixed-camera",
                },
            },
        )

    app = Dash(__name__)
    route_prefix = f"/locust-frame-{id(app)}"
    endpoint = f"locust_frame_image_{id(app)}"

    def serve_frame(index):
        if index < 0 or index >= len(image_paths):
            abort(404)
        return send_file(image_paths[index])

    app.server.add_url_rule(
        f"{route_prefix}/<int:index>", endpoint, serve_frame
    )

    mark_indices = np.unique(
        np.linspace(0, len(frame_values) - 1, min(8, len(frame_values)), dtype=int)
    )
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Img(
                        id="video-frame",
                        src=f"{route_prefix}/0",
                        style={"width": "100%", "height": f"{height}px", "objectFit": "contain"},
                    ),
                    dcc.Graph(
                        id="trajectory-plot",
                        figure=make_figure(0),
                        config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
                        style={"height": f"{height}px", "minWidth": 0},
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1.1fr)",
                    "gap": "12px",
                    "alignItems": "start",
                },
            ),
            dcc.Slider(
                id="frame-slider",
                min=0,
                max=len(frame_values) - 1,
                step=1,
                value=0,
                marks={int(i): str(frame_values[int(i)]) for i in mark_indices},
                updatemode="drag",
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Pre(id="viewer-error", style={"color": "#a00", "whiteSpace": "pre-wrap"}),
        ],
        style={"padding": "12px"},
    )

    @app.callback(
        Output("video-frame", "src"),
        Output("trajectory-plot", "figure"),
        Output("viewer-error", "children"),
        Input("frame-slider", "value"),
    )
    def update_frame(frame_index):
        try:
            frame_index = int(frame_index)
            return f"{route_prefix}/{frame_index}", make_figure(frame_index), ""
        except Exception as exc:
            return f"Frame update failed: {type(exc).__name__}: {exc}"

    return app

if __name__ == '__main__':
    ds_name = f'/bb_kp/20230329/0_6000_5.0Hz'
    ds = load_preprocessed_data(ds_name + '.hdf5')
    img_dir = '/original/20230329/video/'

    app = locust_trajectory_viewer(img_dir, ds, x_var = 'density_voronoi_None', y_var = 'polarization_voronoi_None', z_var = 'ang_speed_spline')
    app.run()
    