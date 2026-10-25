"""Run paired scenarios, save CSV results, summary statistics, and SVG plots."""

from __future__ import annotations

import csv
from dataclasses import asdict
import math
from pathlib import Path
import statistics
import sys

# Ensure src/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from network_simulator import SCENARIOS, simulate
    from ppo_controller import PPOController
    from train_ppo_policy import main as train_policy
except ImportError:
    from src.network_simulator import SCENARIOS, simulate
    from src.ppo_controller import PPOController
    from src.train_ppo_policy import main as train_policy


REPEATS = 10
SEEDS = range(200, 200 + REPEATS)  # held-out evaluation seeds
MODES = ("reno", "heuristic", "ppo")
METRICS = {
    "throughput_mbps": ("Goodput (Mbps)", "higher"),
    "packet_delivery_ratio": ("Unique packet delivery ratio", "higher"),
    "average_delay_ms": ("Mean end-to-end delay (ms)", "lower"),
    "retransmissions": ("Retransmissions (packets)", "lower"),
}


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_svg(metric: str, title: str, summary: list[dict[str, str]]) -> None:
    width, height = 1040, 580
    left, right, top, bottom = 115, 40, 95, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    scenarios = [s.name for s in SCENARIOS]
    selected = [r for r in summary if r["metric"] == metric]
    vals = [float(r["mean"]) for r in selected]
    errs = [float(r["sd"]) for r in selected]
    ymax = max((v + e for v, e in zip(vals, errs)), default=1.0)
    ymax = ymax * 1.16 if ymax > 0 else 1.0
    data = {mode: {r["scenario"]: (float(r["mean"]), float(r["sd"]))
                   for r in selected if r["mode"] == mode} for mode in MODES}
    group = plot_w / max(1, len(scenarios))
    bar_w = min(68, group * 0.28)
    scale = plot_h / ymax
    colors = {"reno": "#8392A5", "heuristic": "#0F9D8A", "ppo": "#D98132"}

    def y(value: float) -> float:
        return top + plot_h - value * scale

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#FFFFFF"/>',
             f'<text x="{left}" y="48" font-family="Arial" font-size="27" font-weight="700" fill="#172B4D">{title}</text>']
    for i in range(6):
        value = ymax * i / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#E5EAF0"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" font-family="Arial" font-size="13" fill="#566579">{value:.2f}</text>')
    for j, scenario in enumerate(scenarios):
        center = left + group * (j + 0.5)
        for series_index, mode in enumerate(MODES):
            value, err = data[mode].get(scenario, (0, 0))
            offset = (series_index - 1) * bar_w * 0.92
            x = center + offset - bar_w / 2
            yy = y(value)
            h = top + plot_h - yy
            parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{colors[mode]}"/>')
            e_top, e_bottom = y(min(ymax, value + err)), y(max(0.0, value - err))
            ex = center + offset
            parts.append(f'<line x1="{ex:.1f}" y1="{e_top:.1f}" x2="{ex:.1f}" y2="{e_bottom:.1f}" stroke="#263648" stroke-width="2"/>')
            parts.append(f'<line x1="{ex-6:.1f}" y1="{e_top:.1f}" x2="{ex+6:.1f}" y2="{e_top:.1f}" stroke="#263648" stroke-width="2"/>')
            parts.append(f'<line x1="{ex-6:.1f}" y1="{e_bottom:.1f}" x2="{ex+6:.1f}" y2="{e_bottom:.1f}" stroke="#263648" stroke-width="2"/>')
        label = scenario.replace("_", " ")
        parts.append(f'<text x="{center:.1f}" y="{height-bottom+28}" text-anchor="middle" font-family="Arial" font-size="14" fill="#34445A">{label}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="14" fill="#566579">Scenario (bars show mean; whiskers show ±1 SD across {REPEATS} seeds)</text>')
    parts.append(f'<text transform="translate(28 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14" fill="#566579">{title}</text>')
    for i, (mode, label) in enumerate(zip(MODES, ("Reno-style", "Rule heuristic", "PPO"))):
        yy = 60 + i * 23
        parts.append(f'<rect x="{width-270}" y="{yy}" width="14" height="14" fill="{colors[mode]}"/><text x="{width-248}" y="{yy+13}" font-family="Arial" font-size="14" fill="#34445A">{label}</text>')
    parts.append('</svg>')
    (FIGURES_DIR / f"{metric}.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    train_policy()
    ppo_policy_file = RESULTS_DIR / "ppo_policy.json"
    ppo_agent = PPOController.load(ppo_policy_file, seed=77)
    raw: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for mode in MODES:
                result = simulate(scenario, mode, seed, ppo_agent=ppo_agent if mode == "ppo" else None)
                raw.append(asdict(result))

    raw_path = RESULTS_DIR / "replicate_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)

    summary: list[dict[str, str]] = []
    for scenario in SCENARIOS:
        for mode in MODES:
            group = [row for row in raw if row["scenario"] == scenario.name and row["mode"] == mode]
            for metric in METRICS:
                m, sd = mean_sd([float(row[metric]) for row in group])
                summary.append({"scenario": scenario.name, "mode": mode, "metric": metric,
                                "mean": f"{m:.6f}", "sd": f"{sd:.6f}",
                                "n": str(len(group)),
                                "completion_rate": f"{sum(bool(row['completed']) for row in group) / len(group):.3f}"})
    with (RESULTS_DIR / "summary_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    for metric, (title, _) in METRICS.items():
        write_svg(metric, title, summary)

    print(f"Completed {len(raw)} held-out evaluation runs ({REPEATS} seeds × {len(SCENARIOS)} scenarios × 3 modes)")
    print("scenario,metric,reno_mean,heuristic_mean,ppo_mean")
    for scenario in SCENARIOS:
        for metric, (_, direction) in METRICS.items():
            vals = {(r["mode"]): float(r["mean"]) for r in summary if r["scenario"] == scenario.name and r["metric"] == metric}
            print(f"{scenario.name},{metric},{vals['reno']:.4f},{vals['heuristic']:.4f},{vals['ppo']:.4f} ({direction} is better)")


if __name__ == "__main__":
    main()
