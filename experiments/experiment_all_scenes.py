# ── Multi-scene experiment: Duke, Washington DC, Boulder ──────────────────────
# Runs all 3 scenes sequentially with identical hyperparameters and output
# structure.  Scenes are loaded/unloaded one at a time (Sionna limitation:
# multiple scenes in same process not supported — NVlabs/sionna#173).

import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

sys.path.append(os.path.abspath('../src'))

try:
    import sionna.rt
except ImportError:
    os.system("pip install sionna-rt")
    import sionna.rt

import gc
import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import mitsuba as mi
import drjit as dr
import torch
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
warnings.filterwarnings("ignore", category=UserWarning, module="jupyter_client")

from sionna.rt import load_scene, Receiver, Camera, PathSolver, AntennaArray
from sionna.rt.antenna_pattern import antenna_pattern_registry

from boresight_pathsolver import create_zone_mask
from multi_tx_optimizer import (
    optimize_multi_tx, setup_bs_transmitters, seed_jammer_positions,
    JammerConfig, compare_multi_tx_performance,
)
from scene_parser import extract_building_info
from shapely.geometry import Polygon as ShapelyPolygon


# ── Scene registry ─────────────────────────────────────────────────────────────
SCENE_CONFIGS = [
    {"scene_name": "duke",         "scene_xml": "../scene/scenes/Duke/scene.xml"},
    {"scene_name": "washington_dc","scene_xml": "../scene/scenes/dupont_circle/scene.xml"},
    {"scene_name": "boulder",      "scene_xml": "../scene/scenes/boulder_open/scene.xml"},
]

OUTPUT_BASE = pathlib.Path("../results/3_scenes_w_jammers")


# ── Shared experiment parameters ───────────────────────────────────────────────
SHAPES = ["square", "circle", "splat", "l_shape"]

SIZE_CONFIGS = [
    {"name": "large",  "size_m": 400.0, "n_bs": 4, "standoff_m": 150.0},
    {"name": "small",  "size_m": 200.0, "n_bs": 2, "standoff_m": 150.0},
]

SAMPLE_COUNTS = [250, 500, 750, 1000]

N_JAMMERS             = 12
OUTSIDE_MARGIN_M      = 200.0
BS_HEIGHT_M           = 50.0
BS_TARGET_Z_M         = 1.5
BS_PROJECT_EDGE       = True
JAM_MIN_BS_DISTANCE_M = 75.0

_base_hparams = dict(
    lds="Halton",
    sampler="rejection",
    sampling_strata="full",
    gamma_db=0.0,
    inside_margin_db=20.0,
    min_sinr_db=20.0,
    lambda_in=3.0,
    lambda_out=15.0,
    lambda_uniform=0.2,
    lambda_min_j=1.0,
    use_gate_logits=True,     # False = all jammers always fully on (useful for debugging coverage)
    lambda_spread=0.0,
    spread_min_dist=0.0,
    soft_mean_weight=6.0,
    soft_mean_in_weight=6.0,
    learning_rate=0.1,
    lr_position=4.0,
    lr_power=0.1,
    noise_power=1e-10,
    lr_scheduler="cosine",
)

MAP_CONFIG = {
    "center":       [0.0, 0.0, 0.0],
    "size":         [1400, 1400],
    "cell_size":    (0.5, 0.5),
    "ground_height": 0.0,
}


# ── Helper functions (defined once, shared across all scenes) ──────────────────

def make_zone_params(shape: str, size_m: float) -> dict:
    if shape == "square":
        s = float(size_m)
        vertices = [(-s, -s), (s, -s), (s, s), (-s, s)]
        return {"center": [0.0, 0.0], "vertices": [(float(x), float(y)) for x, y in vertices]}

    if shape == "circle":
        r = float(size_m)
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        vertices = [(round(r * np.cos(t), 3), round(r * np.sin(t), 3)) for t in theta]
        return {"center": [0.0, 0.0], "vertices": vertices}

    if shape == "splat":
        rng   = np.random.default_rng(seed=42)
        n_pts = 120
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        r     = np.full(n_pts, size_m, dtype=float)
        for freq, (lo_f, hi_f) in [(2, (0.28, 0.44)), (3, (0.19, 0.33)),
                                    (4, (0.11, 0.21)), (5, (0.06, 0.12))]:
            amp   = rng.uniform(lo_f * size_m, hi_f * size_m)
            phase = rng.uniform(0, 2 * np.pi)
            r    += amp * np.cos(freq * theta + phase)
        r = np.clip(r, 0.35 * size_m, 1.65 * size_m)
        vertices = [
            (round(r[i] * np.cos(theta[i]), 2), round(r[i] * np.sin(theta[i]), 2))
            for i in range(n_pts)
        ]
        return {"center": [0.0, 0.0], "vertices": vertices}

    if shape == "l_shape":
        s = float(size_m)
        vertices = [(-s, -s), (s, -s), (s, 0.0), (0.0, 0.0), (0.0, s), (-s, s)]
        return {"center": [0.0, 0.0], "vertices": [(float(x), float(y)) for x, y in vertices]}

    raise ValueError(f"Unknown shape: {shape!r}")


def _remove_existing_bs(scene, max_n_bs=4):
    for idx in range(max_n_bs):
        try:
            scene.remove(f"bs_{idx}")
        except Exception:
            pass


def _seed_jammers_uniform(zone_params, shape, n_jammers=8,
                          standoff_distance=100.0, bs_positions=None,
                          min_bs_distance_m=75.0):
    cx, cy = zone_params.get("center", [0.0, 0.0])

    if shape == "square":
        verts = zone_params["vertices"]
        s = max(abs(float(v[0]) - cx) for v in verts)
        n_per_side = max(1, n_jammers // 4)
        fracs = [(k + 0.5) / n_per_side for k in range(n_per_side)]
        sides = [
            ((cx - s, cy - s), (cx + s, cy - s), ( 0.0, -1.0)),
            ((cx + s, cy - s), (cx + s, cy + s), ( 1.0,  0.0)),
            ((cx + s, cy + s), (cx - s, cy + s), ( 0.0,  1.0)),
            ((cx - s, cy + s), (cx - s, cy - s), (-1.0,  0.0)),
        ]
        positions = []
        for (x0, y0), (x1, y1), (nx, ny) in sides:
            for f in fracs:
                bx = x0 + f * (x1 - x0)
                by = y0 + f * (y1 - y0)
                positions.append([bx + nx * standoff_distance,
                                   by + ny * standoff_distance])
        return positions

    if shape == "circle":
        verts = [(float(v[0]) + cx, float(v[1]) + cy) for v in zone_params["vertices"]]
        poly  = ShapelyPolygon(verts)
        outer = poly.buffer(standoff_distance, join_style="round", resolution=64)
        ring  = outer.exterior
        total = ring.length
        return [
            [ring.interpolate((i + 0.5) / n_jammers * total).x,
             ring.interpolate((i + 0.5) / n_jammers * total).y]
            for i in range(n_jammers)
        ]

    _bs = [[float(p[0]), float(p[1])] for p in bs_positions] if bs_positions else None
    return seed_jammer_positions(
        zone_params,
        n_jammers=n_jammers,
        bs_positions=_bs,
        standoff_distance=standoff_distance,
        min_bs_distance=min_bs_distance_m,
        concave_order=8,
        max_gap_deg=90.0,
        interpolation_factor=2,
        seed=22,
    )


def _extract_stats(stats_dict, tx_configs, use_jam=True):
    out = {}
    for cfg in tx_configs:
        s = stats_dict[cfg.name]
        if use_jam:
            containment = s.get("containment_jam", s.get("containment", {})) or {}
            sinr_in  = s.get("sinr_inside_jam",  s.get("sinr_inside",  {})) or {}
            sinr_out = s.get("sinr_outside_jam", s.get("sinr_outside", {})) or {}
        else:
            containment = s.get("containment", {}) or {}
            sinr_in  = s.get("sinr_inside",  {}) or {}
            sinr_out = s.get("sinr_outside", {}) or {}
        out[cfg.name] = {
            "rho_leak": containment.get("rho_leak"),
            "rho_hole": containment.get("rho_hole"),
            "sinr_inside": {
                "mean_db":   sinr_in.get("sinr_mean_db"),
                "median_db": sinr_in.get("sinr_median_db"),
                "p10_db":    sinr_in.get("sinr_p10_db"),
                "p90_db":    sinr_in.get("sinr_p90_db"),
            },
            "sinr_outside": {
                "mean_db":   sinr_out.get("sinr_mean_db"),
                "median_db": sinr_out.get("sinr_median_db"),
                "p10_db":    sinr_out.get("sinr_p10_db"),
                "p90_db":    sinr_out.get("sinr_p90_db"),
            },
        }
    return out


def _mean_sinr(r, key="with_jammers", region="sinr_inside", stat="mean_db"):
    return np.nanmean([v[region].get(stat, float("nan")) for v in r[key].values()])


def _mean_rho(r, key="with_jammers", metric="rho_leak"):
    vals = [v[metric] for v in r[key].values() if v[metric] is not None]
    return np.mean(vals) if vals else float("nan")


def run_scene_analysis(all_results, scene_name, output_root, shapes, size_configs):
    """Generate per-scene analysis plots and print summary table."""
    if not all_results:
        print(f"  No results for {scene_name}, skipping analysis.")
        return

    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    size_names = [c["name"] for c in size_configs]

    # ── 0. Jamming effectiveness ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    _x = np.arange(len(shapes))
    _w = 0.35
    for ax, region, title in [
        (axes[0], "sinr_inside",  "Mean SINR Inside Zone (dB)"),
        (axes[1], "sinr_outside", "Mean SINR Outside Zone (dB)"),
    ]:
        bs_means, bs_stds, jam_means, jam_stds = [], [], [], []
        for sh in shapes:
            bs_vals  = [_mean_sinr(r, key="bs_only",      region=region)
                        for r in all_results if r["shape"] == sh]
            jam_vals = [_mean_sinr(r, key="with_jammers", region=region)
                        for r in all_results if r["shape"] == sh]
            bs_means.append(np.nanmean(bs_vals));   bs_stds.append(np.nanstd(bs_vals))
            jam_means.append(np.nanmean(jam_vals)); jam_stds.append(np.nanstd(jam_vals))
        ax.bar(_x - _w / 2, bs_means,  _w, yerr=bs_stds,  capsize=4,
               label="BS only",      color="steelblue", alpha=0.85)
        ax.bar(_x + _w / 2, jam_means, _w, yerr=jam_stds, capsize=4,
               label="BS + Jammers", color="tomato",    alpha=0.85)
        ax.set_xticks(_x); ax.set_xticklabels(shapes)
        ax.set_xlabel("Zone Shape"); ax.set_ylabel("Mean SINR (dB)"); ax.set_title(title)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"{scene_name.title()} — Jamming Effectiveness", fontsize=13)
    plt.tight_layout()
    fig.savefig(analysis_dir / "jamming_effectiveness.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── 1. Shape vs. SINR ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, region, title in [
        (axes[0], "sinr_inside",  "Mean SINR Inside Zone (dB)"),
        (axes[1], "sinr_outside", "Mean SINR Outside Zone (dB)"),
    ]:
        by_shape = {sh: [] for sh in shapes}
        for r in all_results:
            by_shape[r["shape"]].append(_mean_sinr(r, region=region))
        means = [np.nanmean(by_shape[sh]) for sh in shapes]
        stds  = [np.nanstd(by_shape[sh])  for sh in shapes]
        ax.bar(shapes, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xlabel("Zone Shape"); ax.set_ylabel("Mean SINR (dB)"); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"{scene_name.title()} — Shape vs. SINR", fontsize=13)
    plt.tight_layout()
    fig.savefig(analysis_dir / "shape_vs_sinr.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── 2. Size vs. SINR ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, region, title in [
        (axes[0], "sinr_inside",  "Mean SINR Inside Zone (dB)"),
        (axes[1], "sinr_outside", "Mean SINR Outside Zone (dB)"),
    ]:
        by_size = {sn: [] for sn in size_names}
        for r in all_results:
            by_size[r["size_name"]].append(_mean_sinr(r, region=region))
        means = [np.nanmean(by_size[sn]) for sn in size_names]
        stds  = [np.nanstd(by_size[sn])  for sn in size_names]
        ax.bar(size_names, means, yerr=stds, capsize=4, color="darkorange", alpha=0.8)
        ax.set_xlabel("Zone Size"); ax.set_ylabel("Mean SINR (dB)"); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"{scene_name.title()} — Size vs. SINR", fontsize=13)
    plt.tight_layout()
    fig.savefig(analysis_dir / "size_vs_sinr.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── 3. Sample count vs. SINR inside ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for shape in shapes:
        by_n = {}
        for r in all_results:
            if r["shape"] == shape:
                by_n.setdefault(r["n_samples"], []).append(
                    _mean_sinr(r, region="sinr_inside"))
        if by_n:
            ns = sorted(by_n)
            ax.plot(ns, [np.nanmean(by_n[n]) for n in ns], marker="o", label=shape)
    ax.set_xlabel("Sample Count"); ax.set_ylabel("Mean Inside SINR (dB)")
    ax.set_title(f"{scene_name.title()} — Sample Count vs. Inside SINR")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(analysis_dir / "samples_vs_sinr.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── 4. Containment metrics ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in [
        (axes[0], "rho_leak", "Leakage Rate (rho_leak)"),
        (axes[1], "rho_hole", "Hole Rate (rho_hole)"),
    ]:
        by_shape = {sh: [] for sh in shapes}
        for r in all_results:
            by_shape[r["shape"]].append(_mean_rho(r, metric=metric))
        means = [np.nanmean(by_shape[sh]) for sh in shapes]
        stds  = [np.nanstd(by_shape[sh])  for sh in shapes]
        ax.bar(shapes, means, yerr=stds, capsize=4, color="tomato", alpha=0.8)
        ax.set_xlabel("Zone Shape"); ax.set_ylabel(metric); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"{scene_name.title()} — Containment Metrics", fontsize=13)
    plt.tight_layout()
    fig.savefig(analysis_dir / "containment_metrics.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── 5. Timing: avg iteration time ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], "warmup_avg_iter_s", "Warm-up Avg Iter Time (s)"),
        (axes[1], "joint_avg_iter_s",  "Joint Avg Iter Time (s)"),
    ]:
        by_shape = {sh: [] for sh in shapes}
        for r in all_results:
            v = r.get("timing", {}).get(key)
            if v is not None:
                by_shape[r["shape"]].append(v)
        means = [np.nanmean(by_shape[sh]) if by_shape[sh] else float("nan") for sh in shapes]
        stds  = [np.nanstd(by_shape[sh])  if by_shape[sh] else 0.0            for sh in shapes]
        ax.bar(shapes, means, yerr=stds, capsize=4, color="mediumseagreen", alpha=0.8)
        ax.set_xlabel("Zone Shape"); ax.set_ylabel("Avg Iter Time (s)"); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"{scene_name.title()} — Iteration Time by Shape", fontsize=13)
    plt.tight_layout()
    fig.savefig(analysis_dir / "timing_by_shape.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── 6. Timing vs. sample count ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], "warmup_avg_iter_s", "Warm-up Avg Iter Time (s)"),
        (axes[1], "joint_avg_iter_s",  "Joint Avg Iter Time (s)"),
    ]:
        for shape in shapes:
            by_n = {}
            for r in all_results:
                if r["shape"] == shape:
                    v = r.get("timing", {}).get(key)
                    if v is not None:
                        by_n.setdefault(r["n_samples"], []).append(v)
            if by_n:
                ns = sorted(by_n)
                ax.plot(ns, [np.nanmean(by_n[n]) for n in ns], marker="o", label=shape)
        ax.set_xlabel("Sample Count"); ax.set_ylabel("Avg Iter Time (s)"); ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle(f"{scene_name.title()} — Iteration Time vs. Sample Count", fontsize=13)
    plt.tight_layout()
    fig.savefig(analysis_dir / "timing_vs_samples.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── 7. Summary table ──────────────────────────────────────────────────────
    print(f"\n{'Run':<35} {'In μ':>7} {'In p10':>7} {'In p90':>7} "
          f"{'Out μ':>7} {'leak':>7} {'hole':>7} {'wu s/it':>8} {'jt s/it':>8} {'tot s':>7}")
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: (x["shape"], x["size_name"], x["n_samples"])):
        tag = f"{r['shape']}_{r['size_name']}_n{r['n_samples']}"
        t   = r.get("timing", {})
        print(
            f"{tag:<35} "
            f"{_mean_sinr(r, region='sinr_inside',  stat='mean_db'):>7.2f} "
            f"{_mean_sinr(r, region='sinr_inside',  stat='p10_db'):>7.2f} "
            f"{_mean_sinr(r, region='sinr_inside',  stat='p90_db'):>7.2f} "
            f"{_mean_sinr(r, region='sinr_outside', stat='mean_db'):>7.2f} "
            f"{_mean_rho(r, metric='rho_leak'):>7.4f} "
            f"{_mean_rho(r, metric='rho_hole'):>7.4f} "
            f"{t.get('warmup_avg_iter_s', float('nan')):>8.2f} "
            f"{t.get('joint_avg_iter_s',  float('nan')):>8.2f} "
            f"{t.get('total_elapsed_s',   float('nan')):>7.1f}"
        )
    print(f"\n  Analysis plots saved to: {analysis_dir.resolve()}")


# ── Main: iterate over scenes ──────────────────────────────────────────────────
for scene_cfg in SCENE_CONFIGS:
    scene_name = scene_cfg["scene_name"]
    scene_xml  = scene_cfg["scene_xml"]
    OUTPUT_ROOT = OUTPUT_BASE / scene_name
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SCENE: {scene_name}  |  {scene_xml}")
    print(f"{'='*70}")

    # ── Load scene ────────────────────────────────────────────────────────────
    scene = load_scene(scene_xml)
    scene.frequency = 3.7e9

    gnb_pattern      = antenna_pattern_registry.get("tr38901")(polarization="V")
    friendly_pattern = antenna_pattern_registry.get("iso")(polarization="V")
    ue_pattern       = antenna_pattern_registry.get("iso")(polarization="V")

    single_element = np.array([[0.0, 0.0, 0.0]])

    scene.tx_array = AntennaArray(
        antenna_pattern=gnb_pattern,
        normalized_positions=single_element.T
    )
    jammer_array = AntennaArray(
        antenna_pattern=friendly_pattern,
        normalized_positions=single_element.T
    )
    scene.rx_array = AntennaArray(
        antenna_pattern=ue_pattern,
        normalized_positions=single_element.T
    )

    rx = Receiver(name="ue", position=[10.0, 0.0, 0.0], display_radius=0.03)
    scene.add(rx)

    for radio_material in scene.radio_materials.values():
        radio_material.scattering_coefficient = 0.4

    print(f"  Scene loaded  |  freq={scene.frequency/1e9} GHz")

    # ── Building polygons (exclude from SINR stats) ───────────────────────────
    _building_info = extract_building_info(scene_xml, verbose=False)
    BUILDING_POLYGONS = []
    for _binfo in _building_info.values():
        try:
            verts_2d = [(float(c[0]), float(c[1])) for c in _binfo["vertices"]]
            _bp = ShapelyPolygon(verts_2d)
            if _bp.is_valid:
                BUILDING_POLYGONS.append(_bp)
        except Exception:
            pass
    print(f"  Building polygons: {len(BUILDING_POLYGONS)}")

    # ── Save map config once per scene ────────────────────────────────────────
    with open(OUTPUT_ROOT / "map_config.json", "w") as _f:
        json.dump(MAP_CONFIG, _f, indent=2)

    total_runs = len(SHAPES) * len(SIZE_CONFIGS) * len(SAMPLE_COUNTS)
    print(f"  Total runs: {total_runs}  |  Output: {OUTPUT_ROOT.resolve()}")

    # ── Experiment loop ───────────────────────────────────────────────────────
    run_index = 0
    for shape in SHAPES:
        for size_cfg in SIZE_CONFIGS:
            for n_samples in SAMPLE_COUNTS:
                run_index += 1
                size_name  = size_cfg["name"]
                size_m     = size_cfg["size_m"]
                n_bs       = size_cfg["n_bs"]
                standoff_m = size_cfg["standoff_m"]
                run_tag    = f"{shape}_{size_name}_n{n_samples}"
                run_dir    = OUTPUT_ROOT / run_tag

                if (run_dir / "results.json").exists():
                    print(f"  [{run_index:>3}/{total_runs}] SKIP  {run_tag}")
                    continue

                print(f"\n  [{run_index:>3}/{total_runs}] START {run_tag}")
                run_dir.mkdir(parents=True, exist_ok=True)

                _remove_existing_bs(scene, max_n_bs=4)

                zone_params = make_zone_params(shape, size_m)
                zone_mask, _, zone_stats = create_zone_mask(
                    map_config=MAP_CONFIG,
                    zone_type="polygon" if "vertices" in zone_params else "box",
                    zone_params=zone_params,
                    target_height=1.5,
                    scene_xml_path=scene_xml,
                    exclude_buildings=True,
                )
                print(f"    Zone cells: {zone_stats['num_cells']}")

                np.save(run_dir / "zone_mask.npy", zone_mask.astype(np.float32))

                tx_configs, bs_positions_xyz = setup_bs_transmitters(
                    scene=scene,
                    zone_params=zone_params,
                    n_bs=n_bs,
                    scene_xml_path=scene_xml,
                    bs_height=BS_HEIGHT_M,
                    target_z=BS_TARGET_Z_M,
                    project_to_edge=BS_PROJECT_EDGE,
                    name_prefix="bs",
                    seed=44,
                )

                jam_positions = _seed_jammers_uniform(
                    zone_params, shape=shape, n_jammers=N_JAMMERS,
                    standoff_distance=standoff_m,
                    bs_positions=bs_positions_xyz,
                    min_bs_distance_m=JAM_MIN_BS_DISTANCE_M,
                )
                jam_configs = [
                    JammerConfig(name=f"jam_{k+1}", initial_power_dbm=20.0,
                                 initial_position=pos)
                    for k, pos in enumerate(jam_positions)
                ]
                print(f"    Jammers: {len(jam_configs)} at {standoff_m:.0f} m standoff")

                _cx, _cy  = zone_params.get("center", [0.0, 0.0])
                _max_xy   = max(max(abs(vx - _cx), abs(vy - _cy))
                                for vx, vy in zone_params["vertices"])
                outside_half = _max_xy + OUTSIDE_MARGIN_M
                _shared = dict(
                    tx_configs=tx_configs,
                    map_config=MAP_CONFIG,
                    scene_xml_path=scene_xml,
                    num_sample_points=n_samples,
                    outside_half_size=outside_half,
                    **_base_hparams,
                )

                # Run 1: BS-only warm-up — inside coverage only
                print("    Run 1: BS warm-up (100 iter, inside coverage only)")
                _warmup_kwargs = {**_shared, "lambda_in": 10.0,
                                  "lambda_out": 0.0, "soft_mean_weight": 0.0}
                result_bs, _ = optimize_multi_tx(
                    scene=scene,
                    jam_configs=None,
                    num_iterations=100,
                    **_warmup_kwargs,
                )
                elapsed_warmup = result_bs["joint"]["elapsed_time_s"]
                n_iter_warmup  = result_bs["joint"]["num_iterations"]
                print(f"    Warm-up: {elapsed_warmup:.1f}s  "
                      f"({elapsed_warmup/n_iter_warmup:.2f}s/iter)")

                # Warm-start Run 2 from Run 1 angles/positions
                for cfg in tx_configs:
                    r   = result_bs[cfg.name]
                    pos = r["final_position"]
                    scene.get(cfg.name).position = mi.Point3f(
                        float(pos[0]), float(pos[1]), float(pos[2])
                    )
                    cfg.initial_azimuth_deg, cfg.initial_elevation_deg = (
                        float(a) for a in r["best_angles"]
                    )

                # Run 2: Joint BS + jammers — BS positions frozen
                print("    Run 2: Joint optimization (100 iter, BS positions frozen)")
                result_jam, jam_scene = optimize_multi_tx(
                    scene=scene,
                    jammer_array=jammer_array,
                    jam_configs=jam_configs,
                    num_iterations=100,
                    freeze_bs=False,
                    **_shared,
                )
                elapsed_joint = result_jam["joint"]["elapsed_time_s"]
                n_iter_joint  = result_jam["joint"]["num_iterations"]
                print(f"    Joint:   {elapsed_joint:.1f}s  "
                      f"({elapsed_joint/n_iter_joint:.2f}s/iter)")

                print("    Evaluating...")
                zone_masks_dict = {cfg.name: zone_mask for cfg in tx_configs}
                fig_map, fig_cdf, stats_combined = compare_multi_tx_performance(
                    scene=scene,
                    tx_configs=tx_configs,
                    multi_result=result_jam,
                    map_config=MAP_CONFIG,
                    zone_masks=zone_masks_dict,
                    noise_power=1e-10,
                    gamma_db=0.0,
                    jam_scene=jam_scene,
                    jammer_configs=jam_configs,
                    building_polygons=BUILDING_POLYGONS,
                    outside_half_size=outside_half,
                    save_arrays_to=run_dir,
                )
                fig_map.savefig(run_dir / "radiomap.png",  dpi=150, bbox_inches="tight")
                fig_cdf.savefig(run_dir / "cdf_sinr.png",  dpi=150, bbox_inches="tight")
                plt.close("all")

                results = {
                    "scene":     scene_name,
                    "shape":     shape,
                    "size_name": size_name,
                    "size_m":    size_m,
                    "n_bs":      n_bs,
                    "n_samples": n_samples,
                    "n_jammers": len(jam_configs),
                    "timing": {
                        "warmup_elapsed_s":  elapsed_warmup,
                        "warmup_n_iter":     n_iter_warmup,
                        "warmup_avg_iter_s": elapsed_warmup / n_iter_warmup,
                        "joint_elapsed_s":   elapsed_joint,
                        "joint_n_iter":      n_iter_joint,
                        "joint_avg_iter_s":  elapsed_joint / n_iter_joint,
                        "total_elapsed_s":   elapsed_warmup + elapsed_joint,
                    },
                    "params": {
                        **_base_hparams,
                        "outside_half_size_m":      outside_half,
                        "num_iterations_warmup":    100,
                        "num_iterations_joint":     100,
                        "warmup_lambda_in":         10.0,
                        "warmup_lambda_out":        0.0,
                        "warmup_soft_mean_weight":  0.0,
                        "bs_height_m":              BS_HEIGHT_M,
                        "bs_target_ue_height_m":    BS_TARGET_Z_M,
                        "bs_project_to_edge":       BS_PROJECT_EDGE,
                        "bs_seed":                  44,
                        "jammer_seed":              22,
                    },
                    "constraints": {
                        "bs_initial_power_dbm":  tx_configs[0].initial_power_dbm,
                        "bs_power_min_dbm":      tx_configs[0].power_dbm_bounds[0],
                        "bs_power_max_dbm":      tx_configs[0].power_dbm_bounds[1],
                        "jam_initial_power_dbm": jam_configs[0].initial_power_dbm,
                        "jam_power_min_dbm":     jam_configs[0].power_dbm_bounds[0],
                        "jam_power_max_dbm":     jam_configs[0].power_dbm_bounds[1],
                        "jam_elevation_bounds":  list(jam_configs[0].elevation_bounds),
                        "jam_standoff_m":        standoff_m,
                        "jam_min_bs_distance_m": JAM_MIN_BS_DISTANCE_M,
                    },
                    "bs_only":      _extract_stats(stats_combined, tx_configs, use_jam=False),
                    "with_jammers": _extract_stats(stats_combined, tx_configs, use_jam=True),
                    "opt_params": {
                        "bs": {
                            cfg.name: {
                                "initial_position":  result_jam[cfg.name]["initial_position"],
                                "initial_angles":    result_jam[cfg.name]["initial_angles"],
                                "initial_power_dbm": result_jam[cfg.name]["initial_power_dbm"],
                                "final_position":    result_jam[cfg.name]["final_position"],
                                "final_angles":      result_jam[cfg.name]["best_angles"],
                                "final_power_dbm":   result_jam[cfg.name]["best_power_dbm"],
                            }
                            for cfg in tx_configs
                        },
                        "jammers": result_jam["joint"].get("jammers", {}),
                    },
                }
                with open(run_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2, default=float)

                print(f"    Saved → {run_dir}")

                del result_bs, result_jam, jam_scene, fig_map, fig_cdf, stats_combined
                gc.collect()
                torch.cuda.empty_cache()
                dr.flush_kernel_cache()
                dr.flush_malloc_cache()

    print(f"\n  Scene {scene_name}: {run_index} runs processed.")

    # ── Per-scene analysis ────────────────────────────────────────────────────
    scene_results = []
    for json_path in sorted(OUTPUT_ROOT.rglob("results.json")):
        with open(json_path) as f:
            scene_results.append(json.load(f))
    print(f"\n  Running analysis on {len(scene_results)} result(s)...")
    run_scene_analysis(scene_results, scene_name, OUTPUT_ROOT, SHAPES, SIZE_CONFIGS)

    # ── Between-scene cleanup (Sionna: one scene per process) ────────────────
    del scene, jammer_array
    gc.collect()
    torch.cuda.empty_cache()
    dr.flush_kernel_cache()
    dr.flush_malloc_cache()
    print(f"  Scene {scene_name} cleaned up.\n")


# ── Cross-scene combined analysis ─────────────────────────────────────────────
print("\n" + "="*70)
print("COMBINED ANALYSIS — all scenes")
print("="*70)

all_results_combined = []
for json_path in sorted(OUTPUT_BASE.rglob("results.json")):
    with open(json_path) as f:
        all_results_combined.append(json.load(f))

print(f"Loaded {len(all_results_combined)} total results across all scenes.")

if all_results_combined:
    combined_dir = OUTPUT_BASE / "combined_analysis"
    combined_dir.mkdir(exist_ok=True)

    scene_names = [sc["scene_name"] for sc in SCENE_CONFIGS]
    colors = ["steelblue", "darkorange", "mediumseagreen"]

    # ── Scene comparison: mean inside/outside SINR ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    _x = np.arange(len(scene_names))
    _w = 0.35
    for ax, region, title in [
        (axes[0], "sinr_inside",  "Mean SINR Inside Zone (dB)"),
        (axes[1], "sinr_outside", "Mean SINR Outside Zone (dB)"),
    ]:
        bs_means, bs_stds, jam_means, jam_stds = [], [], [], []
        for sn in scene_names:
            sub = [r for r in all_results_combined if r["scene"] == sn]
            bs_means.append(np.nanmean([_mean_sinr(r, key="bs_only",      region=region) for r in sub]))
            bs_stds.append( np.nanstd( [_mean_sinr(r, key="bs_only",      region=region) for r in sub]))
            jam_means.append(np.nanmean([_mean_sinr(r, key="with_jammers", region=region) for r in sub]))
            jam_stds.append( np.nanstd( [_mean_sinr(r, key="with_jammers", region=region) for r in sub]))
        ax.bar(_x - _w / 2, bs_means,  _w, yerr=bs_stds,  capsize=4,
               label="BS only",      color="steelblue", alpha=0.85)
        ax.bar(_x + _w / 2, jam_means, _w, yerr=jam_stds, capsize=4,
               label="BS + Jammers", color="tomato",    alpha=0.85)
        ax.set_xticks(_x); ax.set_xticklabels(scene_names)
        ax.set_xlabel("Scene"); ax.set_ylabel("Mean SINR (dB)"); ax.set_title(title)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.suptitle("All Scenes — Jamming Effectiveness Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(combined_dir / "scenes_jamming_effectiveness.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── Shape × scene heatmap: mean inside SINR ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    heat_data = np.full((len(SHAPES), len(scene_names)), np.nan)
    for si, sn in enumerate(scene_names):
        for shi, sh in enumerate(SHAPES):
            vals = [_mean_sinr(r, key="with_jammers", region="sinr_inside")
                    for r in all_results_combined if r["scene"] == sn and r["shape"] == sh]
            if vals:
                heat_data[shi, si] = np.nanmean(vals)
    im = ax.imshow(heat_data, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(scene_names))); ax.set_xticklabels(scene_names)
    ax.set_yticks(range(len(SHAPES)));      ax.set_yticklabels(SHAPES)
    plt.colorbar(im, ax=ax, label="Mean Inside SINR (dB)")
    ax.set_title("Mean Inside SINR (dB) — Shape × Scene (BS+Jammers)")
    for i in range(len(SHAPES)):
        for j in range(len(scene_names)):
            if not np.isnan(heat_data[i, j]):
                ax.text(j, i, f"{heat_data[i, j]:.1f}", ha="center", va="center",
                        fontsize=9, color="black")
    plt.tight_layout()
    fig.savefig(combined_dir / "heatmap_shape_scene.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Scene':<15} {'Shape':<10} {'Size':<8} {'N':>5} "
          f"{'In μ':>7} {'Out μ':>7} {'leak':>7} {'tot s':>7}")
    print("-" * 85)
    for r in sorted(all_results_combined,
                    key=lambda x: (x["scene"], x["shape"], x["size_name"], x["n_samples"])):
        t = r.get("timing", {})
        print(
            f"{r['scene']:<15} {r['shape']:<10} {r['size_name']:<8} {r['n_samples']:>5} "
            f"{_mean_sinr(r, region='sinr_inside',  stat='mean_db'):>7.2f} "
            f"{_mean_sinr(r, region='sinr_outside', stat='mean_db'):>7.2f} "
            f"{_mean_rho(r, metric='rho_leak'):>7.4f} "
            f"{t.get('total_elapsed_s', float('nan')):>7.1f}"
        )

    print(f"\nCombined analysis saved to: {combined_dir.resolve()}")

print("\nAll scenes complete.")
