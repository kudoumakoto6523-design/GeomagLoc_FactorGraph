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


def pipeline(source="uji", data_root="data/raw"):
    n = 11

    map_info, map_mapping, mag_map = _load_map_interfaces(source=source, data_root=data_root)
    for i in range(n):
        templist = []
        geomag_list = []
        stplen_list = []
        heading_angle_list = []
        route = sim.get_true_route(source=source, data_root=data_root)
        init = sim.initialize()
        if init is None:
            init = [0.0, 0.0]
        init = np.asarray(init, dtype=float).reshape(-1)
        if init.size < 2:
            init = np.asarray([0.0, 0.0], dtype=float)
        pos_list = [np.asarray([float(init[0]), float(init[1])], dtype=float).tolist()]
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

                # Use the full accumulated history each iteration (no sliding window).
                stplen_list_using = stplen_list
                heading_angle_list_using = heading_angle_list
                geomag_list_using = geomag_list
                start_idx = 0
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
                    iteration=100,
                    constrain_method="penalty",
                )
                pos_window = [np.asarray(p, dtype=float).reshape(-1)[:2].tolist() for p in pos_window]
                pos_list = pos_list[:start_idx] + pos_window
                templist.clear()
            else:
                continue
