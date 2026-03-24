'''
@ author: kudoumakoto
@ email: kudoumakoto6523@gmail.com
@ description:

This is the pipeline of the geomagnetic location algorithm using factor graph.
The details are as follows:
ATTENTION: This is not a algorithm based on SLAM or its relative algorithms. This is a prebuild map-based
algorithm, using geomagnetic fingerprint.
VERSION DECLARATION: For simplify the algorithm, the motion model will be pedestrian dead reckon (PDR).
The repo is acclaimed to be universal as the code improves in the promising future.

As I cannot draw a factor graph directly, a description can be given as follows:
Connection between two states is consisted by PDR(step length and heading angle), and the observation of geomagnetic
intensity is only related to the current state, which strictly obeys the assumptions in Hidden Markov Model (HMM)

'''

import numpy as np
from pathlib import Path

try:
    from . import simulation as sim
    from . import factor_graph as fg
except ImportError:
    import simulation as sim
    import factor_graph as fg


def _load_map_interfaces(source="uji", data_root="data/raw"):
    map_info = sim.get_map(source=source, data_root=data_root)
    map_mapping = sim.get_map_mapping(geomag_map=map_info, interpolation="bicubic")

    if isinstance(map_info, dict) and "output_preview_npz" in map_info:
        preview = np.load(map_info["output_preview_npz"])
        grid_x = np.asarray(preview["grid_x"], dtype=float)
        grid_y = np.asarray(preview["grid_y"], dtype=float)
        grid_mag = np.asarray(preview["grid_magnitude"], dtype=float)
    elif isinstance(map_info, dict) and map_info.get("grid_array") is not None:
        grid_mag = np.asarray(map_info["grid_array"], dtype=float)
        meta = map_info.get("grid_map_contract", {}).get("meta", {})
        cell_size = float(meta.get("cell_size_m", 1.0) or 1.0)
        origin = meta.get("origin_xy_m", [0.0, 0.0])
        ox = float(origin[0]) if len(origin) > 0 else 0.0
        oy = float(origin[1]) if len(origin) > 1 else 0.0
        rows, cols = grid_mag.shape
        grid_x = ox + np.arange(cols, dtype=float) * cell_size
        grid_y = oy + np.arange(rows, dtype=float) * cell_size
    else:
        raise ValueError("Unsupported map_info format for pipeline map loading.")

    # Keep current torch factor-graph path unchanged; it expects a callable mag_map(position_tensor).
    mag_map = fg.build_mag_map_function(
        mag_map_array=grid_mag,
        x_range=(float(grid_x[0]), float(grid_x[-1])),
        y_range=(float(grid_y[0]), float(grid_y[-1])),
    )
    return map_info, map_mapping, mag_map


def _infer_known_initial_xy(route, source="uji", map_info=None):
    try:
        route_arr = np.asarray(route, dtype=float)
    except Exception:
        return None
    if route_arr.ndim != 2 or route_arr.shape[0] == 0 or route_arr.shape[1] < 2:
        return None

    src = str(source).lower()
    if src == "uji":
        model_path = None
        if isinstance(map_info, dict) and map_info.get("output_model_npz"):
            model_path = Path(map_info["output_model_npz"])
        else:
            candidate = Path("data/processed/uji_mag_model_kriging.npz")
            if candidate.exists():
                model_path = candidate

        if model_path is None or not model_path.exists():
            return None

        try:
            model = np.load(model_path)
            origin_lat = float(model["origin_lat"][0])
            origin_lon = float(model["origin_lon"][0])
            x0, y0 = sim._latlon_to_xy(
                np.asarray([float(route_arr[0, 0])], dtype=float),
                np.asarray([float(route_arr[0, 1])], dtype=float),
                origin_lat,
                origin_lon,
            )
            return [float(x0[0]), float(y0[0])]
        except Exception:
            return None

    # For non-UJI sources, route may already be XY.
    return [float(route_arr[0, 0]), float(route_arr[0, 1])]


def _plot_all_step_losses(loss_histories, output_png, show=False):
    if loss_histories is None:
        return None
    curves = [np.asarray(curve, dtype=float).reshape(-1) for curve in loss_histories if len(curve) > 0]
    if len(curves) == 0:
        return None

    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for loss plotting. Install it with: pip install matplotlib"
        ) from exc

    max_len = max(int(c.size) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, curve in enumerate(curves):
        arr[i, : curve.size] = curve
    x = np.arange(1, max_len + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=140)
    fig.patch.set_facecolor("#f5f7fb")
    ax.set_facecolor("#ffffff")

    line_cmap = plt.get_cmap("viridis")
    for i, curve in enumerate(curves):
        color = line_cmap(0.15 + 0.8 * (i / max(1, len(curves) - 1)))
        ax.plot(np.arange(1, curve.size + 1), curve, color=color, linewidth=1.2, alpha=0.9)

    grad_cmap = plt.get_cmap("YlGnBu")
    q_low = 10.0
    q_high = 90.0
    n_bands = 50
    for k in range(n_bands):
        q0 = q_low + (q_high - q_low) * (k / n_bands)
        q1 = q_low + (q_high - q_low) * ((k + 1) / n_bands)
        y0 = np.nanpercentile(arr, q0, axis=0)
        y1 = np.nanpercentile(arr, q1, axis=0)
        band_color = grad_cmap(0.15 + 0.8 * (k / max(1, n_bands - 1)))
        ax.fill_between(x, y0, y1, color=band_color, alpha=0.32, linewidth=0.0)

    q10 = np.nanpercentile(arr, 10, axis=0)
    q50 = np.nanpercentile(arr, 50, axis=0)
    q90 = np.nanpercentile(arr, 90, axis=0)
    ax.plot(x, q10, color="#1d3557", linewidth=2.0, label="Q10")
    ax.plot(x, q50, color="#e76f51", linewidth=2.6, label="Median")
    ax.plot(x, q90, color="#2a9d8f", linewidth=2.0, label="Q90")

    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size > 0 and (float(np.max(finite)) / max(float(np.min(finite)), 1e-12)) > 100.0:
        ax.set_yscale("log")

    ax.set_title("Factor-Graph Optimization Loss Across All Steps")
    ax.set_xlabel("Iteration (within each optimization step)")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    norm = mpl.colors.Normalize(vmin=q_low, vmax=q_high)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=grad_cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.05)
    cbar.set_label("Gradient Bands (Quantile %)")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return str(output_png)


def pipeline(
    source="uji",
    data_root="data/raw",
    use_magnetic_map=True,
    iteration=100,
    window_size=100,
    initial_point=None,
    use_known_initial_point=True,
    output_loss_plot=None,
    show_loss_plot=False,
):
    n = 1

    if use_magnetic_map:
        map_info, map_mapping, mag_map = _load_map_interfaces(source=source, data_root=data_root)
    else:
        map_info, map_mapping, mag_map = None, None, None

    mode = "factor_graph" if use_magnetic_map else "pdr_only"
    loss_histories = []
    loss_plot_path = None
    for i in range(n):
        templist = []
        geomag_list = []
        stplen_list = []
        heading_angle_list = []
        route = sim.get_true_route(source=source, data_root=data_root)

        if initial_point is not None:
            init = initial_point
        elif use_known_initial_point:
            init = _infer_known_initial_xy(route=route, source=source, map_info=map_info)
            if init is None:
                init = sim.initialize()
        else:
            init = sim.initialize()
        if init is None:
            init = [0.0, 0.0]

        init = np.asarray(init, dtype=float).reshape(-1)
        if init.size < 2:
            init = np.asarray([0.0, 0.0], dtype=float)
        pos_list = [np.asarray([float(init[0]), float(init[1])], dtype=float).tolist()]
        pdr_list = [np.asarray([float(init[0]), float(init[1])], dtype=float).tolist()]
        l = sim.get_test_len(source=source, data_root=data_root)
        for j in range(l):
            mag, acc, gyro = sim.get_sensor(source=source, data_root=data_root)
            templist.append([acc, gyro, mag])
            judge = sim.judge_step(templist)
            if judge:
                stplen = sim.get_step_len(templist)
                heading_angle = sim.get_heading_angle(templist)
                geomag = sim.get_mag()
                stplen_list.append(float(stplen))
                heading_angle_list.append(float(heading_angle))
                geomag_list.append(geomag)

                # Always keep a dead-reckoning baseline for comparison plots.
                pdr_prev = np.asarray(pdr_list[-1], dtype=float).reshape(-1)
                if pdr_prev.size < 2:
                    pdr_prev = np.asarray([0.0, 0.0], dtype=float)
                pdr_next = pdr_prev[:2] + np.asarray(
                    [
                        float(stplen) * np.sin(float(heading_angle)),
                        float(stplen) * np.cos(float(heading_angle)),
                    ],
                    dtype=float,
                )
                pdr_list.append(pdr_next.tolist())

                if not use_magnetic_map:
                    pos_list = list(pdr_list)
                    templist.clear()
                    continue

                # Optimize in a recent sliding window to improve stability and runtime.
                if window_size is None:
                    start_idx = 0
                else:
                    w = max(1, int(window_size))
                    start_idx = max(0, len(stplen_list) - w)
                stplen_list_using = stplen_list[start_idx:]
                heading_angle_list_using = heading_angle_list[start_idx:]
                geomag_list_using = geomag_list[start_idx:]
                start_point = np.asarray(pos_list[start_idx], dtype=float).reshape(-1)
                if start_point.size < 2:
                    start_point = np.asarray([0.0, 0.0], dtype=float)

                parameter_this_iteration = fg.param(mag_map = mag_map,
                                                    starting_point = start_point[:2],
                                                    mag_sensor = geomag_list_using,
                                                    steplength = stplen_list_using,
                                                    heading_angle = heading_angle_list_using
                                                    )

                # TODO
                factor_graph = fg.Factor_Graph(param_this_iteration = parameter_this_iteration)

                pos_window = factor_graph.run(
                    learning_rate=0.01,
                    iteration=iteration,
                    constrain_method="penalty",
                )
                loss_histories.append(list(getattr(factor_graph, "last_loss_history", [])))
                pos_window = [np.asarray(p, dtype=float).reshape(-1)[:2].tolist() for p in pos_window]
                pos_list = pos_list[:start_idx] + pos_window
                templist.clear()
            else:
                continue

    if use_magnetic_map and output_loss_plot:
        loss_plot_path = _plot_all_step_losses(
            loss_histories=loss_histories,
            output_png=output_loss_plot,
            show=show_loss_plot,
        )

    return {
        "mode": mode,
        "map_info": map_info,
        "route": route,
        "positions": pos_list,
        "pdr_positions": pdr_list,
        "steps_detected": len(stplen_list),
        "window_size": None if window_size is None else int(max(1, int(window_size))),
        "loss_histories": loss_histories,
        "loss_plot_path": loss_plot_path,
    }
