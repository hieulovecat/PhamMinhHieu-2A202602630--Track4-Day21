"""Đo độ cao mặt phẳng RANSAC theo distance_threshold (giải thích failure fail_01).

Với mỗi frame và mỗi dt: độ cao mặt phẳng tại gốc sensor (z0 = -d/c) và "độ cao cắt hiệu dụng"
= z0 + dt - z_ground_thật, với z_ground_thật lấy là z0 khi dt = 0.05 (lớp mỏng nhất ~ mặt đường).

    python -m src.plane_offset --data-root data/kitti_mini --out results/plane_offset_kitti.csv
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.obstacle_pipeline import Params, crop_roi, ransac_plane
from starter.datasets import list_frames, load_points

import open3d as o3d


def main() -> None:
    ap = argparse.ArgumentParser(description="Độ cao mặt phẳng RANSAC và độ cao cắt hiệu dụng theo distance_threshold")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--dts", nargs="+", type=float, default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5])
    ap.add_argument("--out", default="results/plane_offset_kitti.csv")
    args = ap.parse_args()

    rows = []
    for f in list_frames(args.data_root):
        roi = crop_roi(load_points(args.data_root, f), Params())
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(roi[:, :3].astype(np.float64)))
        xyz = np.asarray(pcd.voxel_down_sample(Params.voxel_size).points)
        z_ref = None
        for dt in args.dts:
            p = replace(Params(), distance_threshold=dt)
            plane, inl = ransac_plane(xyz, dt, p.ransac_iters, p.seed)
            a, b, c, d = plane
            z0 = -d / c
            z_ref = z0 if z_ref is None else z_ref
            rows.append({"frame": f, "dt": dt, "plane_z_at_sensor": z0,
                         "tilt_deg": float(np.degrees(np.arccos(abs(c)))),
                         "plane_shift_m": z0 - z_ref, "cut_height_m": z0 + dt - z_ref,
                         "ground_ratio": float(inl.mean())})
    df = pd.DataFrame(rows).round(3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby("dt").agg(plane_shift_median=("plane_shift_m", "median"),
                                   plane_shift_max=("plane_shift_m", "max"),
                                   cut_height_median=("cut_height_m", "median"),
                                   cut_height_max=("cut_height_m", "max"),
                                   frames_shift_gt_10cm=("plane_shift_m", lambda s: int((s > 0.10).sum())))
    print(summary.round(3).to_string())
    print(f"-> {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
