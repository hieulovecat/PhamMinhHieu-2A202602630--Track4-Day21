"""Vẽ failure case: so sánh 2 cấu hình pipeline trên cùng 1 object GT (zoom BEV + mặt cắt đứng + ảnh camera).

    python -m src.failure --frame 000011 --gt 0 --a dt=0.15 --b dt=0.5 --name fail_02_dt05_cut_legs
    python -m src.failure --help

--gt là chỉ số object trong label_2 (bỏ DontCare), đúng thứ tự in ra bởi src.obstacle_pipeline.
--a/--b nhận chuỗi "khoá=giá trị" ngăn bởi dấu phẩy, khoá: v (voxel_size), dt (distance_threshold), eps.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.obstacle_pipeline import Params, _cluster_colors, box_bev_corners, evaluate, load, points_in_box, run_pipeline
from starter.projection import project_velo_to_image

KEYS = {"v": "voxel_size", "dt": "distance_threshold", "eps": "eps"}


def parse_cfg(s: str) -> Params:
    kw = {}
    for part in s.split(","):
        k, v = part.split("=")
        kw[KEYS[k.strip()]] = float(v)
    return replace(Params(), **kw)


def panel(axes, fr, obj, p: Params, title: str, zoom: float) -> str:
    res = run_pipeline(fr["points"], p)
    ev = evaluate(res, fr["labels"], fr["calib"], fr["image"].shape, p)
    gi = fr["labels"].index(obj)
    g = ev["gts"][gi]
    down, ground, obs, lab = res["down_xyz"], res["ground_mask"], res["obs_xyz"], res["labels"]
    bev = box_bev_corners(obj, fr["calib"])
    cx, cy = bev.mean(0)
    colors = _cluster_colors(len(res["clusters"]))

    # các cluster có điểm trong box GT
    in_box = points_in_box(obs, obj, fr["calib"], margin_xy=0.3, z_low=-0.3, z_high=0.3)
    ks = np.unique(lab[in_box & (lab >= 0)])

    def near(xyz):
        return (np.abs(xyz[:, 0] - cx) < zoom) & (np.abs(xyz[:, 1] - cy) < zoom)

    # BEV zoom
    ax = axes[0]
    m = near(down) & ground
    ax.scatter(down[m, 0], down[m, 1], s=4, c="#c8c8c8", label="ground (RANSAC)")
    m = near(obs) & (lab < 0)
    ax.scatter(obs[m, 0], obs[m, 1], s=6, c="k", marker="x", label="noise DBSCAN")
    for k in ks:
        m = lab == k
        ax.scatter(obs[m, 0], obs[m, 1], s=6, color=colors[k], label=f"cluster {k} ({m.sum()} điểm)")
        c = res["clusters"][k]
        ax.add_patch(plt.Rectangle(c["min"][:2], *(c["max"][:2] - c["min"][:2]), fill=False, ec=colors[k], lw=1.5))
    m = near(obs) & (lab >= 0) & ~np.isin(lab, ks)
    ax.scatter(obs[m, 0], obs[m, 1], s=3, c="#8888ff", alpha=0.5, label="cluster khác")
    ax.add_patch(plt.Polygon(bev, fill=False, ec="lime", lw=2.5, label=f"GT {obj.type}"))
    ax.set_xlim(cx - zoom, cx + zoom)
    ax.set_ylim(cy - zoom, cy + zoom)
    ax.set_aspect("equal")
    ax.set_title(f"{title}: BEV", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

    # mặt cắt đứng: khoảng cách ngang từ sensor vs độ cao z, chỉ lấy điểm quanh box
    ax = axes[1]
    body = points_in_box(down, obj, fr["calib"], margin_xy=0.3, z_low=-0.5, z_high=0.3)
    r = np.hypot(down[:, 0], down[:, 1])
    ax.scatter(r[body & ground], down[body & ground, 2], s=10, c="#999999", label="bị coi là ground")
    ax.scatter(r[body & ~ground], down[body & ~ground, 2], s=10, c="#d62728", label="giữ lại (non-ground)")
    a, b, c_, d = res["plane"]
    rr = np.linspace(r[body].min() - 1, r[body].max() + 1, 2) if body.any() else np.array([0, 1])
    # độ cao mặt phẳng ước lượng tại tâm box (gần đúng: bỏ qua nghiêng theo phương ngang)
    z_plane = -(a * cx + b * cy + d) / c_
    for off, ls in ((0, "-"), (p.distance_threshold, "--")):
        ax.plot(rr, [z_plane + off] * 2, "b" + ls, lw=1)
    ax.set_xlabel("khoảng cách ngang tới sensor (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"mặt cắt đứng (xanh: mặt phẳng ground, nét đứt: +dt={p.distance_threshold} m)", fontsize=9)
    ax.legend(fontsize=7)

    body_obs = points_in_box(obs, obj, fr["calib"], margin_xy=0.1, z_low=0.1)
    n_in_cluster = int((body_obs & (lab >= 0)).sum())
    verdict = (f"{title}: thân vật còn {g['n_body_kept']}/{g['n_body']} điểm sau ground, "
               f"{n_in_cluster} điểm thuộc cluster -> coverage {'ĐẠT' if g['covered'] else 'TRƯỢT'}; "
               f"instance: {'tách riêng' if g['detected'] else ('DÍNH VÀO VẬT KHÁC' if g['merged'] else 'MẤT')} "
               f"({len(ks)} cluster chạm box); {ev['n_clusters']} cluster/frame")
    axes[0].set_xlabel(verdict, fontsize=8)
    return verdict, res, ks, colors


def main() -> None:
    ap = argparse.ArgumentParser(description="Ảnh failure case: 2 cấu hình x (BEV zoom, mặt cắt đứng) + ảnh camera")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", required=True)
    ap.add_argument("--gt", type=int, required=True, help="chỉ số object trong label (bỏ DontCare)")
    ap.add_argument("--a", default="dt=0.15", help="cấu hình A, ví dụ 'dt=0.15' hoặc 'v=0.1,eps=0.5'")
    ap.add_argument("--b", default="dt=0.5", help="cấu hình B")
    ap.add_argument("--zoom", type=float, default=6.0, help="nửa cạnh vùng zoom BEV (m)")
    ap.add_argument("--name", required=True, help="tên file, nên bắt đầu bằng fail_XX_")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    fr = load(args.data_root, args.frame)
    obj = fr["labels"][args.gt]
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 0.9])
    verdicts = []
    for col, cfg in enumerate((args.a, args.b)):
        p = parse_cfg(cfg)
        v, res, ks, colors = panel([fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])], fr, obj, p,
                                   f"[{cfg}]", args.zoom)
        verdicts.append(v)
        # ảnh camera crop quanh 2D box, điểm cluster chạm box tô màu theo cluster
        img = fr["image"].copy()
        obs, lab = res["obs_xyz"], res["labels"]
        sel = np.isin(lab, ks)
        uv, _, m = project_velo_to_image(obs[sel], fr["calib"], img.shape)
        cols = (colors[lab[sel][m]] * 255).astype(int)
        for (u, vv), c in zip(uv.astype(int), cols):
            cv2.circle(img, (int(u), int(vv)), 2, (int(c[2]), int(c[1]), int(c[0])), -1)
        x1, y1, x2, y2 = (int(round(x)) for x in obj.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        pad = 120
        crop = img[max(0, y1 - pad):y2 + pad, max(0, x1 - 2 * pad):x2 + 2 * pad]
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title("camera: điểm của các cluster chạm box GT (xanh lá = 2D box label)", fontsize=9)
        ax.axis("off")
    rng = float(np.hypot(obj.location[0], obj.location[2]))
    fig.suptitle(f"{args.data_root} / {args.frame} / GT #{args.gt} {obj.type} cách {rng:.1f} m, "
                 f"cao {obj.dimensions[0]:.2f} m", fontsize=12)
    fig.tight_layout()
    out = Path(args.out_dir) / f"{args.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    for v in verdicts:
        print(v)
    print(f"-> {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
