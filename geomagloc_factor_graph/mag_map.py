"""
Continuous magnetic-map interface:
- world <-> grid coordinate transforms
- bicubic interpolation query
- spatial gradient query
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ContinuousMagMap:
    def __init__(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        grid_magnitude: np.ndarray,
        *,
        method: str = "bicubic",
        clip_to_bounds: bool = True,
    ) -> None:
        self.grid_x = np.asarray(grid_x, dtype=float).reshape(-1)
        self.grid_y = np.asarray(grid_y, dtype=float).reshape(-1)
        self.grid_magnitude = np.asarray(grid_magnitude, dtype=float)
        self.method = str(method).lower()
        self.clip_to_bounds = bool(clip_to_bounds)

        if self.grid_x.size < 2 or self.grid_y.size < 2:
            raise ValueError("grid_x and grid_y must each contain at least 2 points.")
        if self.grid_magnitude.shape != (self.grid_y.size, self.grid_x.size):
            raise ValueError(
                "grid_magnitude shape must be (len(grid_y), len(grid_x))."
            )
        if not np.all(np.diff(self.grid_x) > 0.0):
            raise ValueError("grid_x must be strictly increasing.")
        if not np.all(np.diff(self.grid_y) > 0.0):
            raise ValueError("grid_y must be strictly increasing.")

        self.x_min = float(self.grid_x[0])
        self.x_max = float(self.grid_x[-1])
        self.y_min = float(self.grid_y[0])
        self.y_max = float(self.grid_y[-1])
        self.width = int(self.grid_x.size)
        self.height = int(self.grid_y.size)
        self.dx = (self.x_max - self.x_min) / float(self.width - 1)
        self.dy = (self.y_max - self.y_min) / float(self.height - 1)

        if self.method == "bicubic":
            self._build_bicubic_spline()
        elif self.method == "bilinear":
            self._spline = None
        else:
            raise ValueError("Unsupported interpolation method. Use 'bicubic' or 'bilinear'.")

    def _build_bicubic_spline(self) -> None:
        try:
            from scipy.interpolate import RectBivariateSpline
        except ImportError as exc:
            raise ImportError(
                "scipy is required for bicubic interpolation. Install with: pip install scipy"
            ) from exc

        # RectBivariateSpline expects z.shape == (len(x), len(y)).
        # Our matrix convention is z[row(y), col(x)], so transpose here.
        kx = min(3, self.grid_x.size - 1)
        ky = min(3, self.grid_y.size - 1)
        self._spline = RectBivariateSpline(
            self.grid_x,
            self.grid_y,
            self.grid_magnitude.T,
            kx=kx,
            ky=ky,
            s=0.0,
        )

    @classmethod
    def from_preview_npz(
        cls,
        npz_path: str | Path,
        *,
        method: str = "bicubic",
        clip_to_bounds: bool = True,
    ) -> "ContinuousMagMap":
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Preview grid file not found: {npz_path}")
        data = np.load(npz_path)
        return cls(
            grid_x=np.asarray(data["grid_x"], dtype=float),
            grid_y=np.asarray(data["grid_y"], dtype=float),
            grid_magnitude=np.asarray(data["grid_magnitude"], dtype=float),
            method=method,
            clip_to_bounds=clip_to_bounds,
        )

    @classmethod
    def from_map_info(
        cls,
        map_info: dict[str, Any] | str | Path,
        *,
        method: str = "bicubic",
        clip_to_bounds: bool = True,
    ) -> "ContinuousMagMap":
        if isinstance(map_info, (str, Path)):
            return cls.from_preview_npz(
                map_info,
                method=method,
                clip_to_bounds=clip_to_bounds,
            )

        if not isinstance(map_info, dict):
            raise TypeError("map_info must be a dict, path string, or Path.")

        if "output_preview_npz" in map_info:
            return cls.from_preview_npz(
                map_info["output_preview_npz"],
                method=method,
                clip_to_bounds=clip_to_bounds,
            )

        grid_array = map_info.get("grid_array")
        if grid_array is None:
            raise ValueError(
                "map_info must contain `output_preview_npz` or `grid_array`."
            )
        z = np.asarray(grid_array, dtype=float)
        if z.ndim != 2 or z.size == 0:
            raise ValueError("grid_array must be a non-empty 2D matrix.")

        meta = map_info.get("grid_map_contract", {}).get("meta", {})
        cell_size = float(meta.get("cell_size_m", 1.0) or 1.0)
        origin = meta.get("origin_xy_m", [0.0, 0.0])
        origin_x = float(origin[0]) if len(origin) > 0 else 0.0
        origin_y = float(origin[1]) if len(origin) > 1 else 0.0
        rows, cols = z.shape
        grid_x = origin_x + np.arange(cols, dtype=float) * cell_size
        grid_y = origin_y + np.arange(rows, dtype=float) * cell_size
        return cls(
            grid_x=grid_x,
            grid_y=grid_y,
            grid_magnitude=z,
            method=method,
            clip_to_bounds=clip_to_bounds,
        )

    def world_to_grid(self, x: float | np.ndarray, y: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        col = (x_arr - self.x_min) / (self.dx + 1e-12)
        row = (y_arr - self.y_min) / (self.dy + 1e-12)
        return col, row

    def grid_to_world(self, col: float | np.ndarray, row: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col_arr = np.asarray(col, dtype=float)
        row_arr = np.asarray(row, dtype=float)
        x = self.x_min + col_arr * self.dx
        y = self.y_min + row_arr * self.dy
        return x, y

    def _clip_xy(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.clip_to_bounds:
            return x, y
        return (
            np.clip(x, self.x_min, self.x_max),
            np.clip(y, self.y_min, self.y_max),
        )

    def query(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray | float:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_arr, y_arr = np.broadcast_arrays(x_arr, y_arr)
        xq, yq = self._clip_xy(x_arr.reshape(-1), y_arr.reshape(-1))

        if self.method == "bicubic":
            vals = self._spline.ev(xq, yq)
        else:
            vals = self._query_bilinear(xq, yq)

        vals = vals.reshape(x_arr.shape)
        return float(vals.item()) if vals.ndim == 0 else vals

    def query_with_gradient(
        self, x: float | np.ndarray, y: float | np.ndarray
    ) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_arr, y_arr = np.broadcast_arrays(x_arr, y_arr)
        xq, yq = self._clip_xy(x_arr.reshape(-1), y_arr.reshape(-1))

        if self.method == "bicubic":
            vals = self._spline.ev(xq, yq)
            dfdx = self._spline.ev(xq, yq, dx=1, dy=0)
            dfdy = self._spline.ev(xq, yq, dx=0, dy=1)
        else:
            vals, dfdx, dfdy = self._query_bilinear_with_gradient(xq, yq)

        vals = vals.reshape(x_arr.shape)
        dfdx = dfdx.reshape(x_arr.shape)
        dfdy = dfdy.reshape(x_arr.shape)
        if vals.ndim == 0:
            return float(vals.item()), float(dfdx.item()), float(dfdy.item())
        return vals, dfdx, dfdy

    def _query_bilinear(self, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
        vals, _, _ = self._query_bilinear_with_gradient(xq, yq)
        return vals

    def _query_bilinear_with_gradient(self, xq: np.ndarray, yq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        col, row = self.world_to_grid(xq, yq)
        x0 = np.floor(col).astype(int)
        y0 = np.floor(row).astype(int)
        x0 = np.clip(x0, 0, self.width - 2)
        y0 = np.clip(y0, 0, self.height - 2)
        x1 = x0 + 1
        y1 = y0 + 1

        tx = col - x0
        ty = row - y0

        f00 = self.grid_magnitude[y0, x0]
        f01 = self.grid_magnitude[y1, x0]
        f10 = self.grid_magnitude[y0, x1]
        f11 = self.grid_magnitude[y1, x1]

        vals = (
            (1.0 - tx) * (1.0 - ty) * f00
            + tx * (1.0 - ty) * f10
            + (1.0 - tx) * ty * f01
            + tx * ty * f11
        )
        dfdx = (((1.0 - ty) * (f10 - f00)) + (ty * (f11 - f01))) / (self.dx + 1e-12)
        dfdy = (((1.0 - tx) * (f01 - f00)) + (tx * (f11 - f10))) / (self.dy + 1e-12)
        return vals, dfdx, dfdy

    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray | float:
        return self.query(x, y)
