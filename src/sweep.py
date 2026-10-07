"""Sweep tham số của pipeline obstacle (thay đổi MỘT tham số mỗi lần, các tham số khác giữ baseline).

    python -m src.sweep --data-root data/kitti_mini --tag kitti
    python -m src.sweep --data-root data/nuscenes_mini_subset --tag nusc --sweeps baseline distance_threshold
    python -m src.sweep --help

Ghi ra:
    results/obstacle_sweep_<tag>.csv          mỗi dòng = 1 cấu hình, metric gộp trên mọi frame
    results/obstacle_sweep_frames_<tag>.csv   mỗi dòng = 1 cấu hình x 1 frame
    results/obstacle_gt_detail_<tag>.csv      mỗi dòng = 1 cấu hình x 1 object GT (dùng cho failure analysis)
    results/latency_<tag>.csv                 latency p50/p95 mỗi cấu hình (bỏ lần chạy đầu, lặp --reps lần)
    results/figures/sweep_<tham số>_<tag>.png
"""
from __future__ import annotations

import argparse
import sys
import platform
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.obstacle_pipeline import GROUPS, Params, evaluate, load, run_pipeline
from starter.datasets import list_frames

SWEEPS = {
    "baseline": ("voxel_size", [Params.voxel_size]),
    "distance_threshold": ("distance_threshold", [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]),
    "voxel_size": ("voxel_size", [0.05, 0.1, 0.2, 0.3, 0.4]),
    "eps": ("eps", [0.3, 0.5, 0.8, 1.2]),
}


def aggregate(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out = {"n_frames": len(df), "clusters_mean": df["n_clusters"].mean(),
           "precision_fov": df["n_hit_fov"].sum() / max(df["n_clusters_fov"].sum(), 1),
           "nearest_obstacle_m_median": df["nearest_obstacle_m"].median(),
           "ground_ratio_mean": df["ground_ratio"].mean(), "plane_tilt_max_deg": df["plane_tilt_deg"].max(),
           "total_ms_mean": df["total_ms"].mean()}
    for g in GROUPS + ["all"]:
        n = df[f"n_gt_{g}"].sum()
        out[f"n_gt_{g}"] = int(n)
        out[f"recall_{g}"] = df[f"tp_{g}"].sum() / n if n else np.nan
        out[f"coverage_{g}"] = df[f"covered_{g}"].sum() / n if n else np.nan
        out[f"merged_{g}"] = int(df[f"merged_{g}"].sum())
        b = df[f"body_pts_{g}"].sum()
        out[f"retention_{g}"] = df[f"body_kept_{g}"].sum() / b if b else np.nan
    return out


def measure_latency(points: np.ndarray, params: list[Params], reps: int) -> list[dict]:
    """Đo latency xen kẽ: mỗi vòng chạy lần lượt MỌI cấu hình 1 lần, lặp reps+1 vòng, bỏ vòng đầu (warm-up).
    Xen kẽ để nhiễu của máy (tiến trình nền, nhiệt độ CPU) chia đều cho mọi cấu hình, thay vì dồn vào
    cấu hình nào đang chạy đúng lúc máy bận."""
    times = [[] for _ in params]
    for r in range(reps + 1):
        for i, p in enumerate(params):
            t = run_pipeline(points, p)["timing"]
            if r > 0:
                times[i].append(t)
    out = []
    for ts in times:
        tot = np.array([t["total_ms"] for t in ts])
        d = {"reps": reps, "p50_ms": np.percentile(tot, 50), "p95_ms": np.percentile(tot, 95)}
        for k in ("crop_ms", "voxel_ms", "ground_ms", "cluster_ms", "box_ms"):
            d[f"{k}_p50"] = float(np.median([t[k] for t in ts]))
        out.append(d)
    return out


def plot_sweep(df: pd.DataFrame, param: str, out_path: Path, tag: str) -> None:
    d = df[df["sweep"] == param].sort_values(param)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for g, m, col in zip(GROUPS, ["o", "s", "^"], ["C0", "C1", "C2"]):
        n = int(d[f"n_gt_{g}"].iloc[0])
        axes[0].plot(d[param], d[f"recall_{g}"], marker=m, color=col, label=f"{g} recall (n={n})")
        axes[0].plot(d[param], d[f"coverage_{g}"], marker=m, color=col, ls="--", alpha=0.6, label=f"{g} coverage")
        axes[1].plot(d[param], d[f"retention_{g}"], marker=m, color=col, label=g)
    axes[0].set_title("Recall (nét liền, purity>=0.5) và coverage (nét đứt)")
    axes[1].set_title("% điểm thân vật còn lại sau tách mặt đất")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(0, 1.05)
    ax2 = axes[2]
    ax2.plot(d[param], d["clusters_mean"], "k-o", label="số cluster / frame")
    ax2.set_ylabel("số cluster / frame")
    ax3 = ax2.twinx()
    ax3.errorbar(d[param], d["p50_ms"], yerr=[np.zeros(len(d)), d["p95_ms"] - d["p50_ms"]], fmt="r-s",
                 capsize=4, label="latency p50 (thanh = p95)")
    ax3.set_ylabel("ms / frame", color="r")
    ax2.set_title("Số cluster và latency")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8)
    for ax in axes:
        ax.set_xlabel(param)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].legend()
    fig.suptitle(f"Sweep {param} ({tag}); các tham số khác giữ baseline {asdict(Params())}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep voxel_size / distance_threshold / eps cho pipeline obstacle, "
                                             "ghi CSV + biểu đồ")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--tag", default="kitti", help="hậu tố tên file kết quả")
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--frames", nargs="*", default=None, help="mặc định: mọi frame trong data-root")
    ap.add_argument("--latency-frame", default=None, help="frame dùng đo latency (mặc định: frame đầu tiên)")
    ap.add_argument("--reps", type=int, default=20, help="số lần lặp đo latency (sau khi bỏ lần warm-up)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    frames = args.frames or list_frames(args.data_root)
    data = {f: load(args.data_root, f) for f in frames}
    lat_frame = args.latency_frame or frames[0]
    configs = []
    for name in args.sweeps:
        field, values = SWEEPS[name]
        configs += [(name, replace(Params(), **{field: v})) for v in values]

    summary, per_frame, per_gt, latency = [], [], [], []
    for name, p in configs:
        rows = []
        for f in frames:
            fr = data[f]
            res = run_pipeline(fr["points"], p)
            ev = evaluate(res, fr["labels"], fr["calib"], fr["image"].shape, p)
            gts = ev.pop("gts")
            row = {"sweep": name, **asdict(p), "frame": f, **ev, "total_ms": res["timing"]["total_ms"]}
            rows.append(row)
            per_gt += [{"sweep": name, **asdict(p), "frame": f, **g} for g in gts]
        per_frame += rows
        summary.append({"sweep": name, **asdict(p), **aggregate(rows)})
        s = summary[-1]
        print(f"[{name:<18}] v={p.voxel_size:<5} dt={p.distance_threshold:<5} eps={p.eps:<4} "
              f"recall car={s['recall_Car']:.2f} ped={s['recall_Pedestrian']:.2f} cyc={s['recall_Cyclist']:.2f} "
              f"cover ped={s['coverage_Pedestrian']:.2f} ret ped={s['retention_Pedestrian']:.2f} "
              f"clusters={s['clusters_mean']:.1f} prec_fov={s['precision_fov']:.2f}", flush=True)

    print(f"đo latency trên frame {lat_frame}: {args.reps} vòng x {len(configs)} cấu hình ...", flush=True)
    lats = measure_latency(data[lat_frame]["points"], [p for _, p in configs], args.reps)
    for (name, p), lat, s in zip(configs, lats, summary):
        latency.append({"sweep": name, **asdict(p), "frame": lat_frame, **lat,
                        "cpu": platform.processor(), "python": platform.python_version()})
        s["p50_ms"], s["p95_ms"] = lat["p50_ms"], lat["p95_ms"]
        print(f"[{name:<18}] v={p.voxel_size:<5} dt={p.distance_threshold:<5} eps={p.eps:<4} "
              f"p50={lat['p50_ms']:.1f}ms p95={lat['p95_ms']:.1f}ms", flush=True)

    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary).round(4)
    df.to_csv(out / f"obstacle_sweep_{args.tag}.csv", index=False)
    pd.DataFrame(per_frame).round(4).to_csv(out / f"obstacle_sweep_frames_{args.tag}.csv", index=False)
    pd.DataFrame(per_gt).round(3).to_csv(out / f"obstacle_gt_detail_{args.tag}.csv", index=False)
    pd.DataFrame(latency).round(3).to_csv(out / f"latency_{args.tag}.csv", index=False)
    for name in args.sweeps:
        if name != "baseline":
            plot_sweep(df, name, out / "figures" / f"sweep_{name}_{args.tag}.png", args.tag)
    print(f"-> {out}/obstacle_sweep_{args.tag}.csv, latency_{args.tag}.csv, figures/sweep_*_{args.tag}.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
