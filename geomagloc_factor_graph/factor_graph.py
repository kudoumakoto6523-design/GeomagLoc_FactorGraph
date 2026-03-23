"""
@ author: kudoumakoto6523
@ date: 2026-3-22
@ description: This file is used for a definition of a factor graph optimization problem
"""

from dataclasses import dataclass
import math
import numpy as np
import torch
from livelossplot import PlotLosses
from tqdm import trange


@dataclass
class param:
    starting_point: np.array
    mag_map: object
    steplength: list
    heading_angle: list
    mag_sensor: list


def build_mag_map_function(mag_map_array, x_range, y_range):
    """
    This function returns a callable mag_map(position).

    Parameters
    ----------
    mag_map_array:
        2D numpy array or torch tensor.
        Shape: [H, W]
        It stores the geomagnetic intensity on grid points.

    x_range:
        tuple/list like (x_min, x_max)
        Physical range covered by the map in x direction.

    y_range:
        tuple/list like (y_min, y_max)
        Physical range covered by the map in y direction.

    Returns
    -------
    mag_map:
        a function, usage:
            value = mag_map(position)

        where:
            position is torch tensor with shape [2]
            position[0] = x
            position[1] = y

    Notes
    -----
    - This function uses bilinear interpolation.
    - Input position should be torch tensor if you want gradient to flow.
    - If the point is outside the map range, it is clamped to boundary.
    """

    if isinstance(mag_map_array, np.ndarray):
        mag_map_tensor = torch.tensor(mag_map_array, dtype=torch.float32)
    else:
        mag_map_tensor = mag_map_array.to(torch.float32)

    H, W = mag_map_tensor.shape

    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])

    dx = (x_max - x_min) / (W - 1)
    dy = (y_max - y_min) / (H - 1)

    def mag_map(position):
        """
        position: torch tensor, shape [2]
        position[0] = x
        position[1] = y
        """
        x = position[0]
        y = position[1]

        # convert physical coordinate -> grid coordinate
        gx = (x - x_min) / dx
        gy = (y - y_min) / dy

        # clamp inside valid range
        gx = torch.clamp(gx, 0.0, W - 1.0)
        gy = torch.clamp(gy, 0.0, H - 1.0)

        x0 = torch.floor(gx).long()
        x1 = torch.clamp(x0 + 1, max=W - 1)
        y0 = torch.floor(gy).long()
        y1 = torch.clamp(y0 + 1, max=H - 1)

        # interpolation weights
        wx = gx - x0.float()
        wy = gy - y0.float()

        # four neighboring values
        q00 = mag_map_tensor[y0, x0]
        q01 = mag_map_tensor[y0, x1]
        q10 = mag_map_tensor[y1, x0]
        q11 = mag_map_tensor[y1, x1]

        # bilinear interpolation
        value = (
            (1 - wx) * (1 - wy) * q00 +
            wx * (1 - wy) * q01 +
            (1 - wx) * wy * q10 +
            wx * wy * q11
        )

        return value

    return mag_map


def _api_factor_graph_contrain_penalty(constrain_func, target, delta, weight=1e4):
    """
    Standard Quadratic Penalty: lambda * max(0, |h(x)-target| - delta)^2

    constrain_func: this must be a function
    """
    def penalty_wrapper(X):
        error = torch.abs(constrain_func(X) - target)
        violation = torch.clamp(error - delta, min=0)
        return weight * torch.sum(violation ** 2)
    return penalty_wrapper


def _api_factor_graph_contrain_log(constrain_func, target, delta, t=10):
    """
    Log-Barrier Method: - (1/t) * log(delta - |h(x)-target|)

    constrain_func: this must be a function
    """
    def log_wrapper(X):
        error = torch.abs(constrain_func(X) - target)
        gap = delta - error
        gap = torch.clamp(gap, min=1e-12)
        return -(1.0 / t) * torch.sum(torch.log(gap))
    return log_wrapper


def _api_factor_graph_contrain_lag(constrain_func, target, delta, multiplier, mu=1e3):
    """
    Augmented Lagrangian

    constrain_func: this must be a function
    """
    def lag_wrapper(X):
        error = torch.abs(constrain_func(X) - target)
        violation = torch.clamp(error - delta, min=0)
        return torch.sum(multiplier * violation + (mu / 2) * violation ** 2)
    return lag_wrapper


def default_Q1_window(start_point, steplen_list, heading_angle_list, mag_map, mag_sensor_list, param_w, param_v):
    """
    Windowed magnetic loss.

    Parameters
    ----------
    start_point:
        np.array([x, y])

    steplen_list:
        list/array, window size = K

    heading_angle_list:
        list/array, window size = K

    mag_map:
        this must be a function:
            mag_map(position) -> scalar magnetic value

    mag_sensor_list:
        list/array, window size = K

    param_w:
        torch tensor, shape [K]

    param_v:
        torch tensor, shape [K]

    Returns
    -------
    total magnetic loss over the whole window
    """

    current_pos = torch.tensor(start_point, dtype=torch.float32)
    total_loss = torch.tensor(0.0, dtype=torch.float32)

    for i in range(len(steplen_list)):
        corrected_step = torch.tensor(float(steplen_list[i]), dtype=torch.float32) + (2 / math.pi) * torch.atan(param_w[i])
        corrected_angle = torch.tensor(float(heading_angle_list[i]), dtype=torch.float32) + (2 / math.pi) * torch.atan(param_v[i])

        move_vec = corrected_step * torch.stack([
            torch.sin(corrected_angle),
            torch.cos(corrected_angle)
        ])

        current_pos = current_pos + move_vec

        mag_pred = mag_map(current_pos)
        mag_true = torch.tensor(float(mag_sensor_list[i]), dtype=torch.float32)

        total_loss = total_loss + torch.abs(mag_pred - mag_true)

    return total_loss


# TO BE EVALUATED

# TODO

def default_Q2(param_w, target=0.1):
    """
    param_w can be a vector
    """
    return torch.sum((2 / math.pi) * torch.atan(param_w) * target)


def default_Q3(param_v, target=0.1):
    """
    param_v can be a vector
    """
    return torch.sum((2 / math.pi) * torch.atan(param_v) * target)


def target_func(param_this_iteration, Q1=default_Q1_window, Q2=default_Q2, Q3=default_Q3):
    """
    This returns a function:
        combined_objective(param_w, param_v)

    where param_w and param_v are vectors on the whole window.
    """
    def combined_objective(param_w, param_v):
        return (
            Q1(
                param_this_iteration.starting_point,
                param_this_iteration.steplength,
                param_this_iteration.heading_angle,
                param_this_iteration.mag_map,
                param_this_iteration.mag_sensor,
                param_w,
                param_v
            )
            + Q2(param_w)
            + Q3(param_v)
        )
    return combined_objective


def constrain(param_w, param_v):
    """
    param_w, param_v can be vectors
    """
    return (2 / math.pi) * torch.atan(param_w) + (2 / math.pi) * torch.atan(param_v)


class Factor_Graph:

    def __init__(self, param_this_iteration: param, constrain = constrain,target_func=target_func, optimizer="Adam"):
        """
        constrain: this must be a function
        target_func: this must be a function generator
        """
        self.constrain = constrain
        self.param_this_iteration = param_this_iteration
        self.target_func_builder = target_func
        self.optimizer = optimizer

    def optimization(self, learning_rate, iteration, constrain_method="penalty"):
        """
        penalty method: "penalty"
        Log-Barrier Method: "log"
        Augmented Lagrangian Method: "lag"
        """

        window_size = len(self.param_this_iteration.steplength)

        param_w = torch.zeros(window_size, dtype=torch.float32, requires_grad=True)
        param_v = torch.zeros(window_size, dtype=torch.float32, requires_grad=True)

        base_target_func = self.target_func_builder(self.param_this_iteration)

        def constrain_func(_X):
            return self.constrain(param_w, param_v)

        if constrain_method == "penalty":
            constrain_term = _api_factor_graph_contrain_penalty(
                constrain_func=constrain_func,
                target=0.0,
                delta=1.0,
                weight=1e4
            )
        elif constrain_method == "log":
            constrain_term = _api_factor_graph_contrain_log(
                constrain_func=constrain_func,
                target=0.0,
                delta=1.0,
                t=10
            )
        elif constrain_method == "lag":
            multiplier = torch.zeros(window_size, dtype=torch.float32)
            constrain_term = _api_factor_graph_contrain_lag(
                constrain_func=constrain_func,
                target=0.0,
                delta=1.0,
                multiplier=multiplier,
                mu=1e3
            )
        else:
            raise ValueError("constrain_method must be 'penalty', 'log', or 'lag'")

        optimizer = torch.optim.Adam([param_w, param_v], lr=learning_rate)

        plotlosses = PlotLosses()
        pbar = trange(iteration, desc="Optimizing")

        for step in pbar:
            optimizer.zero_grad()

            loss_target = base_target_func(param_w, param_v)
            loss_constrain = constrain_term(None)
            loss = loss_target + loss_constrain

            loss.backward()
            optimizer.step()

            print(f"loss: {loss.item()}")
            plotlosses.update({'loss': loss.item()})
            plotlosses.send()

        print(f"optimizer {self.optimizer} finished its iteration")
        print(f"optimized parameters:")
        print(f"param_w: {param_w.detach().numpy()}")
        print(f"param_v: {param_v.detach().numpy()}")

        start_pos = torch.tensor(self.param_this_iteration.starting_point, dtype=torch.float32)
        current_pos = start_pos.clone()
        pos_list = [current_pos.detach().numpy()]

        for i in range(window_size):
            corrected_step = torch.tensor(float(self.param_this_iteration.steplength[i]), dtype=torch.float32) + (2 / math.pi) * torch.atan(param_w[i])
            corrected_angle = torch.tensor(float(self.param_this_iteration.heading_angle[i]), dtype=torch.float32) + (2 / math.pi) * torch.atan(param_v[i])

            move_vec = corrected_step * torch.stack([
                torch.sin(corrected_angle),
                torch.cos(corrected_angle)
            ])

            current_pos = current_pos + move_vec
            pos_list.append(current_pos.detach().numpy())

        print("Optimized positions in this window:")
        for p in pos_list:
            print(p)

        return pos_list

    def run(self, learning_rate=0.01, iteration=100, constrain_method="penalty"):
        """
        Call this from the class item.
        """
        return self.optimization(
            learning_rate=learning_rate,
            iteration=iteration,
            constrain_method=constrain_method
        )