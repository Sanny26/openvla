#!/usr/bin/env python3
"""
compute_action_distribution_stats.py

Given an action sampling JSON, compute per-DoF mean and standard deviation
for each timestep at a specified temperature (and optional do_sample filter).

Expected JSON structure (per section):
{
  "temp_...": {
    "temperature": float,
    "do_sample": bool,
    "task_id": int,
    "task_description": str,
    "num_episodes": int,
    "success_rate": float,
    "action_distributions": {
      "0": [ [a_00,...,a_0D], [a_10,...,a_1D], ... ],  # samples for timestep 0
      "1": [ ... ],
      ...
    }
  }
}

This script aggregates across the list of samples for each timestep and computes
mean and std per DoF. Results are printed and can optionally be saved to CSV or JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def find_sections(
    data: Dict[str, Any], temperature: float, do_sample: bool | None
) -> List[Tuple[str, Dict[str, Any]]]:
    """Return all sections matching temperature (and do_sample if provided)."""
    out = []
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        if v.get("temperature") != temperature:
            continue
        if do_sample is not None and v.get("do_sample") is not do_sample:
            continue
        out.append((k, v))
    return out


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    m = sum(values) / n
    # population std by default; switch to sample std (n-1) if n>1
    var = 0.0
    if n == 1:
        var = 0.0
    else:
        var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(var)


def compute_stats_for_section(section: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-DoF mean/std for each timestep in a section."""
    ad = section.get("action_distributions", {})
    if not isinstance(ad, dict):
        raise ValueError("action_distributions must be a dict of timestep -> list of samples")

    # Timesteps keys may be strings; sort by integer timestep
    timestep_keys = sorted(ad.keys(), key=lambda k: int(k))

    results: Dict[str, Any] = {
        "temperature": section.get("temperature"),
        "do_sample": section.get("do_sample"),
        "task_id": section.get("task_id"),
        "task_description": section.get("task_description"),
        "num_episodes": section.get("num_episodes"),
        "success_rate": section.get("success_rate"),
        "timestep_stats": {},  # timestep -> { mean: [...], std: [...], num_samples: int }
    }

    for t_key in timestep_keys:
        samples = ad.get(t_key, [])
        if not samples:
            results["timestep_stats"][t_key] = {
                "mean": [],
                "std": [],
                "num_samples": 0,
            }
            continue

        # infer DoF dimension from first sample
        dof = len(samples[0])
        # transpose samples to per-DoF lists
        per_dof: List[List[float]] = [[] for _ in range(dof)]
        for row in samples:
            # tolerate ragged rows; only up to min length
            L = min(len(row), dof)
            for i in range(L):
                per_dof[i].append(float(row[i]))

        means: List[float] = []
        stds: List[float] = []
        for arr in per_dof:
            m, s = mean_std(arr)
            means.append(m)
            stds.append(s)

        results["timestep_stats"][t_key] = {
            "mean": means,
            "std": stds,
            "num_samples": len(samples),
        }

    return results


def save_csv(stats: Dict[str, Any], out_path: Path) -> None:
    """Save per-timestep stats to CSV: columns = timestep, dof, mean, std, num_samples"""
    import csv

    rows = []
    for t_key, t_stats in stats.get("timestep_stats", {}).items():
        means = t_stats.get("mean", [])
        stds = t_stats.get("std", [])
        num = t_stats.get("num_samples", 0)
        for dof_idx, (m, s) in enumerate(zip(means, stds)):
            rows.append({
                "timestep": int(t_key),
                "dof": dof_idx,
                "mean": m,
                "std": s,
                "num_samples": num,
            })

    rows.sort(key=lambda r: (r["timestep"], r["dof"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestep", "dof", "mean", "std", "num_samples"])
        writer.writeheader()
        writer.writerows(rows)


def save_json(stats: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(stats, f, indent=2)


def print_brief(stats: Dict[str, Any], max_steps: int = 10) -> None:
    print("=== Action Distribution Stats ===")
    print(f"Temperature:  {stats.get('temperature')}")
    print(f"Do sample:    {stats.get('do_sample')}")
    print(f"Task:         {stats.get('task_description')}")
    print(f"Success rate: {stats.get('success_rate')}")

    t_keys = sorted(stats.get("timestep_stats", {}).keys(), key=lambda k: int(k))
    if not t_keys:
        print("No timestep stats available.")
        return

    first_key = t_keys[0]
    dof = len(stats["timestep_stats"][first_key].get("mean", []))
    dof_names = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']

    print(f"Timesteps: {len(t_keys)} | DoF: {dof}")
    print("Showing first {} timesteps:".format(min(max_steps, len(t_keys))))
    for t_key in t_keys[:max_steps]:
        t = int(t_key)
        ts = stats["timestep_stats"][t_key]
        means = ts["mean"]
        stds = ts["std"]
        num = ts["num_samples"]
        parts = []
        for i, (m, s) in enumerate(zip(means, stds)):
            name = dof_names[i] if i < len(dof_names) else f"dof{i}"
            parts.append(f"{name}: mean={m:.4f}, std={s:.4f}")
        print(f"t={t:03d} (n={num}): " + " | ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute per-timestep action distribution stats")
    ap.add_argument(
        "--json_file",
        type=Path,
        default=Path("/data/user/san/openvla/experiments/robot/libero/tmp/action_sampling.json"),
        help="Path to action sampling JSON",
    )
    ap.add_argument("--temperature", type=float, required=True, help="Temperature to analyze")
    ap.add_argument(
        "--do_sample",
        type=str,
        default=None,
        choices=[None, "true", "false"],
        help="Optional: filter by do_sample (true/false)",
    )
    ap.add_argument("--save_csv", type=Path, default=None, help="Optional path to save CSV")
    ap.add_argument("--save_json", type=Path, default=None, help="Optional path to save JSON of stats")
    ap.add_argument("--max_print_steps", type=int, default=10, help="How many timesteps to print")

    args = ap.parse_args()

    do_sample = None
    if isinstance(args.do_sample, str):
        if args.do_sample.lower() == "true":
            do_sample = True
        elif args.do_sample.lower() == "false":
            do_sample = False
        else:
            do_sample = None

    data = load_json(args.json_file)
    sections = find_sections(data, args.temperature, do_sample)
    if not sections:
        raise SystemExit(
            f"No sections found for temperature={args.temperature}"
            + (" and do_sample=" + str(do_sample) if do_sample is not None else "")
        )

    # If multiple sections match (e.g., different tasks), compute and output each
    for sec_key, sec in sections:
        print(f"\n-- Section: {sec_key} --")
        stats = compute_stats_for_section(sec)
        print_brief(stats, max_steps=args.max_print_steps)

        if args.save_csv:
            # if multiple sections, annotate filename with section key
            out_csv = args.save_csv
            if len(sections) > 1:
                out_csv = out_csv.with_name(out_csv.stem + f"_{sec_key}" + out_csv.suffix)
            save_csv(stats, out_csv)
            print(f"Saved CSV: {out_csv}")

        if args.save_json:
            out_json = args.save_json
            if len(sections) > 1:
                out_json = out_json.with_name(out_json.stem + f"_{sec_key}" + out_json.suffix)
            save_json(stats, out_json)
            print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()

