"""Topic D: phát hiện vật cản từ LiDAR không dùng deep learning.

Pipeline: crop ROI -> voxel downsample -> RANSAC ground plane -> DBSCAN -> AABB mỗi cluster.
Đánh giá: so khớp cluster với GT box 3D trong label_2 (recall theo class, precision trong FOV camera,
tỉ lệ điểm của vật còn giữ lại sau khi tách mặt đất).

Chạy demo trên 1 frame (ảnh trước/sau từng bước + overlay lên camera + BEV occupancy):
    python -m src.obstacle_pipeline --data-root data/kitti_mini --frame 000011
    python -m src.obstacle_pipeline --data-root data/kitti_mini --frame 000011 --distance-threshold 0.3
    python -m src.obstacle_pipeline --help

Tham khảo API Open3D: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
(voxel_down_sample, segment_plane, cluster_dbscan).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from starter.datasets import dataset_type, load_frame
from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import box3d_corners_cam, project_velo_to_image, velo_to_cam

# Gom class KITTI/nuScenes về 3 nhóm để báo recall; class khác chỉ tính vào nhóm "all".
CLASS_GROUP = {
    "Car": "Car", "Van": "Car", "Truck": "Car", "Tram": "Car", "Bus": "Car",
    "Pedestrian": "Pedestrian", "Person_sitting": "Pedestrian",
    "Cyclist": "Cyclist", "Bicycle": "Cyclist", "Motorcycle": "Cyclist",
}
GROUPS = ["Car", "Pedestrian", "Cyclist"]


@dataclass
class Params:
    voxel_size: float = 0.1          # m, cạnh voxel khi downsample
    distance_threshold: float = 0.15  # m, điểm cách mặt phẳng RANSAC <= ngưỡng này bị coi là mặt đất
    eps: float = 0.5                  # m, bán kính láng giềng của DBSCAN
    min_points: int = 5               # số điểm tối thiểu của DBSCAN (core point)
    ransac_iters: int = 200
    min_range: float = 2.0            # m, bỏ điểm quá gần sensor (thân xe ego của nuScenes)
    max_range: float = 40.0           # m, vùng quan tâm của robot/xe chạy chậm
    seed: int = 0


# ----------------------------------------------------------------------------- pipeline

def crop_roi(points: np.ndarray, p: Params) -> np.ndarray:
    """Bỏ NaN/Inf và giữ điểm có khoảng cách ngang trong [min_range, max_range]."""
    pts = points[np.isfinite(points[:, :3]).all(axis=1)]
    r = np.linalg.norm(pts[:, :2], axis=1)
    return pts[(r >= p.min_range) & (r <= p.max_range)]


def ransac_plane(xyz: np.ndarray, distance_threshold: float, iters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC mặt phẳng (cùng thuật toán với open3d segment_plane, ransac_n=3) nhưng chạy 1 luồng với
    RNG có seed -> kết quả lặp lại được. segment_plane của Open3D 0.20 chạy song song nên dù đã gọi
    o3d.utility.random.seed vẫn có frame (KITTI 000016) ra mặt phẳng khác nhau giữa các lần chạy.

    Trả về (plane [a, b, c, d] với ax+by+cz+d=0, mask inlier)."""
    rng = np.random.default_rng(seed)
    best_plane, best_count = np.array([0.0, 0.0, 1.0, 0.0]), -1
    for _ in range(iters):
        a, b, c = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:   # 3 điểm thẳng hàng
            continue
        n /= norm
        count = int((np.abs(xyz @ n - n @ a) <= distance_threshold).sum())
        if count > best_count:
            best_plane, best_count = np.append(n, -n @ a), count
    inliers = np.abs(xyz @ best_plane[:3] + best_plane[3]) <= distance_threshold
    return best_plane, inliers


def run_pipeline(points: np.ndarray, p: Params) -> dict:
    """points (N, >=3) trong LiDAR frame. Trả về dict các kết quả trung gian + thời gian từng bước (ms)."""
    t = {}
    t0 = time.perf_counter()
    roi = crop_roi(points, p)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(roi[:, :3].astype(np.float64)))
    t["crop_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    down = pcd.voxel_down_sample(p.voxel_size)
    t["voxel_ms"] = (time.perf_counter() - t0) * 1e3
    xyz = np.asarray(down.points)

    t0 = time.perf_counter()
    plane, ground = ransac_plane(xyz, p.distance_threshold, p.ransac_iters, p.seed)
    inliers = np.flatnonzero(ground)
    t["ground_ms"] = (time.perf_counter() - t0) * 1e3
    normal = np.asarray(plane[:3]) / np.linalg.norm(plane[:3])
    tilt_deg = float(np.degrees(np.arccos(abs(normal[2]))))  # 0 = mặt phẳng nằm ngang

    t0 = time.perf_counter()
    obstacle = down.select_by_index(np.asarray(inliers, dtype=int), invert=True)
    labels = np.asarray(obstacle.cluster_dbscan(eps=p.eps, min_points=p.min_points, print_progress=False))
    t["cluster_ms"] = (time.perf_counter() - t0) * 1e3
    obs_xyz = np.asarray(obstacle.points)

    t0 = time.perf_counter()
    clusters = []
    for k in range(labels.max() + 1 if len(labels) else 0):
        c = obs_xyz[labels == k]
        clusters.append({
            "id": k, "n": len(c), "min": c.min(0), "max": c.max(0), "center": c.mean(0),
            "nearest_m": float(np.linalg.norm(c[:, :2], axis=1).min()),
        })
    t["box_ms"] = (time.perf_counter() - t0) * 1e3
    t["total_ms"] = sum(t.values())

    return {"roi": roi, "down_xyz": xyz, "ground_mask": ground, "plane": np.asarray(plane), "tilt_deg": tilt_deg,
            "obs_xyz": obs_xyz, "labels": labels, "clusters": clusters, "timing": t}


# ----------------------------------------------------------------------------- GT matching

def points_in_box(points_xyz: np.ndarray, obj: KittiObject, calib: KittiCalib,
                  margin_xy: float = 0.2, z_low: float = -0.2, z_high: float = 0.2) -> np.ndarray:
    """Mask điểm (LiDAR frame) nằm trong box 3D của label (box ở rectified camera frame).

    Chiều cao tính từ đáy box: giữ điểm có độ cao trong [z_low, h + z_high]. Dùng z_low > 0
    để chỉ lấy phần "thân vật" (không lẫn mặt đường nằm ngay dưới chân vật)."""
    pc = velo_to_cam(points_xyz, calib)
    h, w, l = obj.dimensions
    d = pc - obj.location
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    lx = c * d[:, 0] - s * d[:, 2]      # R^T * d, R quay quanh trục y camera
    lz = s * d[:, 0] + c * d[:, 2]
    height = -d[:, 1]                   # camera y hướng xuống -> độ cao so với đáy box
    return ((np.abs(lx) <= l / 2 + margin_xy) & (np.abs(lz) <= w / 2 + margin_xy)
            & (height >= z_low) & (height <= h + z_high))


def evaluate(res: dict, labels: list[KittiObject], calib: KittiCalib, image_shape, p: Params,
             min_gt_points: int = 10, purity_thr: float = 0.5) -> dict:
    """So khớp cluster với GT.

    - GT hợp lệ: thuộc nhóm Car/Pedestrian/Cyclist, tâm trong [min_range, max_range] và có
      >= min_gt_points điểm thân vật (cao hơn đáy box 0.1 m) trong point cloud đã downsample.
    - GT được phát hiện (recall): có ít nhất 1 cluster mà >= purity_thr điểm của cluster nằm trong box GT.
    - GT bị "nuốt" (merged): có điểm thuộc cluster nhưng không cluster nào đủ purity (dính vào tường, cột...).
    - GT được phủ (covered): >= 50% điểm thân vật thuộc một cluster bất kỳ (metric theo góc nhìn tránh vật cản).
    - retention: tỉ lệ điểm thân vật còn lại sau khi tách mặt đất.
    - precision_fov: tỉ lệ cluster có tâm nằm trong ảnh camera khớp được với một box GT (mọi class).
      Đây là cận dưới vì KITTI không gán nhãn tường, cột, cây...
    """
    down, ground = res["down_xyz"], res["ground_mask"]
    obs, lab = res["obs_xyz"], res["labels"]
    n_cl = len(res["clusters"])

    gts = []
    cluster_hit = np.zeros(n_cl, dtype=bool)
    for obj in labels:
        if obj.dimensions[0] <= 0:
            continue
        rng = float(np.hypot(obj.location[0], obj.location[2]))
        body_down = points_in_box(down, obj, calib, margin_xy=0.1, z_low=0.1)
        in_obs = points_in_box(obs, obj, calib, margin_xy=0.3, z_low=-0.3, z_high=0.3)
        purity_ok = False
        touched = False
        for k in np.unique(lab[in_obs & (lab >= 0)]):
            touched = True
            ck = lab == k
            purity = (in_obs & ck).sum() / ck.sum()
            if purity >= purity_thr:
                purity_ok = True
                cluster_hit[k] = True
        group = CLASS_GROUP.get(obj.type)
        n_body = int(body_down.sum())
        # coverage (góc nhìn robot): >= 50% điểm thân vật nằm trong một cluster bất kỳ (kể cả cluster
        # dính với vật khác) -> vùng đó vẫn được đánh dấu là vật cản.
        body_obs = points_in_box(obs, obj, calib, margin_xy=0.1, z_low=0.1)
        covered = n_body > 0 and (body_obs & (lab >= 0)).sum() >= 0.5 * n_body
        valid = group is not None and p.min_range <= rng <= p.max_range and n_body >= min_gt_points
        gts.append({"type": obj.type, "group": group, "range_m": rng, "n_body": n_body,
                    "n_body_kept": int((body_down & ~ground).sum()), "valid": valid,
                    "detected": purity_ok, "merged": touched and not purity_ok, "covered": bool(covered)})

    # precision trong FOV camera
    in_fov = np.zeros(n_cl, dtype=bool)
    if n_cl:
        centers = np.array([c["center"] for c in res["clusters"]])
        _, _, in_fov = project_velo_to_image(centers, calib, image_shape)
    n_fov = int(in_fov.sum())

    out = {"n_clusters": n_cl, "n_clusters_fov": n_fov, "n_hit_fov": int((cluster_hit & in_fov).sum()),
           "precision_fov": float((cluster_hit & in_fov).sum() / n_fov) if n_fov else np.nan,
           "nearest_obstacle_m": min((c["nearest_m"] for c in res["clusters"]), default=np.nan),
           "plane_tilt_deg": res["tilt_deg"], "n_points_down": len(down), "ground_ratio": float(ground.mean())}
    for g in GROUPS + ["all"]:
        sel = [x for x in gts if x["valid"] and (g == "all" or x["group"] == g)]
        out[f"n_gt_{g}"] = len(sel)
        out[f"tp_{g}"] = sum(x["detected"] for x in sel)
        out[f"merged_{g}"] = sum(x["merged"] for x in sel)
        out[f"covered_{g}"] = sum(x["covered"] for x in sel)
        out[f"body_pts_{g}"] = sum(x["n_body"] for x in sel)
        out[f"body_kept_{g}"] = sum(x["n_body_kept"] for x in sel)
    out["gts"] = gts
    return out


# ----------------------------------------------------------------------------- vẽ hình

def box_bev_corners(obj: KittiObject, calib: KittiCalib) -> np.ndarray:
    """4 góc đáy của box GT, đổi về LiDAR frame, lấy (x, y) để vẽ BEV."""
    corners_cam = box3d_corners_cam(obj)[:4]
    T_velo_cam = np.linalg.inv(calib.T_cam_velo)
    pts = (T_velo_cam @ np.hstack([corners_cam, np.ones((4, 1))]).T).T
    return pts[:, :2]


def _cluster_colors(n: int) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.uniform(0.15, 0.95, size=(max(n, 1), 3))


def draw_bev_steps(fr: dict, res: dict, p: Params, out_path: Path, title: str = "") -> None:
    """4 panel BEV: điểm gốc -> sau voxel -> ground/non-ground -> cluster + AABB + GT."""
    roi, down, ground = res["roi"], res["down_xyz"], res["ground_mask"]
    obs, lab, clusters = res["obs_xyz"], res["labels"], res["clusters"]
    lim = p.max_range
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5))
    s = 0.3
    axes[0].scatter(roi[:, 0], roi[:, 1], s=s, c=roi[:, 2], cmap="viridis", vmin=-2.5, vmax=1)
    axes[0].set_title(f"1. ROI gốc: {len(roi)} điểm")
    axes[1].scatter(down[:, 0], down[:, 1], s=s, c=down[:, 2], cmap="viridis", vmin=-2.5, vmax=1)
    axes[1].set_title(f"2. Voxel {p.voxel_size} m: {len(down)} điểm")
    axes[2].scatter(down[ground, 0], down[ground, 1], s=s, c="#bbbbbb", label=f"ground ({ground.sum()})")
    axes[2].scatter(down[~ground, 0], down[~ground, 1], s=s, c="#d62728", label=f"non-ground ({(~ground).sum()})")
    axes[2].set_title(f"3. RANSAC dt={p.distance_threshold} m (tilt {res['tilt_deg']:.1f}°)")
    axes[2].legend(loc="upper right", markerscale=10)
    colors = _cluster_colors(len(clusters))
    noise = lab < 0
    axes[3].scatter(obs[noise, 0], obs[noise, 1], s=s, c="#cccccc")
    if len(clusters):
        axes[3].scatter(obs[~noise, 0], obs[~noise, 1], s=s, c=colors[lab[~noise]])
    for c in clusters:
        (x0, y0), (x1, y1) = c["min"][:2], c["max"][:2]
        axes[3].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="k", lw=0.6))
    for obj in fr["labels"]:
        bev = box_bev_corners(obj, fr["calib"])
        axes[3].add_patch(plt.Polygon(bev, fill=False, ec="lime", lw=1.8))
        axes[3].text(bev[:, 0].mean(), bev[:, 1].max() + 0.5, obj.type, color="green", fontsize=7)
    axes[3].set_title(f"4. DBSCAN eps={p.eps}: {len(clusters)} cluster (đen=AABB, xanh lá=GT)")
    for ax in axes:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.plot(0, 0, "k^", ms=8)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def draw_camera_overlay(fr: dict, res: dict, out_path: Path) -> None:
    """Chiếu điểm của từng cluster lên ảnh (mỗi cluster một màu), vẽ 2D box GT màu xanh lá."""
    img = fr["image"].copy()
    obs, lab = res["obs_xyz"], res["labels"]
    keep = lab >= 0
    uv, _, mask = project_velo_to_image(obs[keep], fr["calib"], img.shape)
    colors = (_cluster_colors(len(res["clusters"]))[lab[keep][mask]] * 255).astype(int)
    for (u, v), c in zip(uv.astype(int), colors):
        cv2.circle(img, (int(u), int(v)), 2, (int(c[2]), int(c[1]), int(c[0])), -1)
    for obj in fr["labels"]:
        x1, y1, x2, y2 = (int(round(x)) for x in obj.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, obj.type, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def draw_occupancy(res: dict, p: Params, out_path: Path, cell: float = 0.2) -> np.ndarray:
    """BEV occupancy grid 2D: ô bị chiếm nếu có điểm non-ground thuộc một cluster."""
    n = int(2 * p.max_range / cell)
    grid = np.zeros((n, n), dtype=np.uint8)
    obs = res["obs_xyz"][res["labels"] >= 0]
    ij = np.floor((obs[:, :2] + p.max_range) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    grid[ij[ok, 1], ij[ok, 0]] = 1
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid, origin="lower", cmap="Greys", extent=[-p.max_range, p.max_range, -p.max_range, p.max_range])
    ax.plot(0, 0, "r^", ms=8)
    ax.set_title(f"BEV occupancy {cell} m/ô: {int(grid.sum())} ô bị chiếm")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return grid


def load(data_root: str, frame: str) -> dict:
    kwargs = {"use_ego_motion": True} if dataset_type(data_root) == "nuscenes" else {}
    return load_frame(data_root, frame, **kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Topic D: voxel -> RANSAC ground -> DBSCAN obstacle trên 1 frame, "
                                             "lưu ảnh từng bước + overlay camera + occupancy và in metric so với GT")
    ap.add_argument("--data-root", default="data/kitti_mini", help="data/kitti_mini, data/synthetic hoặc data/nuscenes_mini_subset")
    ap.add_argument("--frame", default="000011")
    ap.add_argument("--voxel-size", type=float, default=Params.voxel_size)
    ap.add_argument("--distance-threshold", type=float, default=Params.distance_threshold)
    ap.add_argument("--eps", type=float, default=Params.eps)
    ap.add_argument("--min-points", type=int, default=Params.min_points)
    ap.add_argument("--max-range", type=float, default=Params.max_range)
    ap.add_argument("--seed", type=int, default=Params.seed)
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--prefix", default="demo", help="tiền tố tên file ảnh, ví dụ fail_01")
    args = ap.parse_args()

    p = Params(voxel_size=args.voxel_size, distance_threshold=args.distance_threshold, eps=args.eps,
               min_points=args.min_points, max_range=args.max_range, seed=args.seed)
    fr = load(args.data_root, args.frame)
    res = run_pipeline(fr["points"], p)
    ev = evaluate(res, fr["labels"], fr["calib"], fr["image"].shape, p)

    tag = f"{args.prefix}_{args.frame}_v{p.voxel_size}_dt{p.distance_threshold}_eps{p.eps}"
    out = Path(args.out_dir)
    draw_bev_steps(fr, res, p, out / f"{tag}_bev_steps.png", title=f"{args.data_root} / {args.frame}  {asdict(p)}")
    draw_camera_overlay(fr, res, out / f"{tag}_camera.png")
    draw_occupancy(res, p, out / f"{tag}_occupancy.png")

    t = res["timing"]
    print(f"points roi={len(res['roi'])} down={len(res['down_xyz'])} ground={res['ground_mask'].sum()} "
          f"tilt={res['tilt_deg']:.2f}° clusters={ev['n_clusters']} (fov {ev['n_clusters_fov']}) "
          f"nearest={ev['nearest_obstacle_m']:.2f} m  time={t['total_ms']:.1f} ms")
    for g in GROUPS:
        print(f"  {g:<10} recall {ev[f'tp_{g}']}/{ev[f'n_gt_{g}']}  merged {ev[f'merged_{g}']}  "
              f"covered {ev[f'covered_{g}']}/{ev[f'n_gt_{g}']}  "
              f"body kept {ev[f'body_kept_{g}']}/{ev[f'body_pts_{g}']}")
    for x in ev["gts"]:
        print(f"    GT {x['type']:<11} r={x['range_m']:5.1f} m body={x['n_body']:4d} kept={x['n_body_kept']:4d} "
              f"valid={x['valid']!s:<5} detected={x['detected']!s:<5} merged={x['merged']!s:<5} covered={x['covered']}")
    print(f"  precision_fov={ev['precision_fov']:.2f}  -> {out / tag}_*.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
