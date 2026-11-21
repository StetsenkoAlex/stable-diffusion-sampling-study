"""Reproduce the Stable Diffusion 1.5 study tables from measured runs and scores."""

import argparse
import csv
import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np


SAMPLERS = ("ddim", "euler", "dpmpp_2m")
STEPS = (5, 10, 20, 30, 50)
GENERATION_SEEDS = (42, 1234, 2026)
PROMPTS = tuple(f"P{i:02d}" for i in range(1, 11))
STEP_COMPARISONS = ((5, 10), (10, 20), (20, 30), (30, 50), (20, 50))
BOOTSTRAP_SEED = 20261117
N_BOOTSTRAPS = 10_000


class Results(NamedTuple):
    per_prompt: list
    conditions: list
    step_comparisons: list
    sampler_comparisons: list


def summarize(clip, pairs):
    """Use one matched set of ten prompts and the same bootstrap draws for all contrasts."""
    images = defaultdict(list)
    distances = defaultdict(list)
    for row in clip:
        images[row["sampler"], row["steps"], row["prompt_id"]].append(row)
    for row in pairs:
        distances[row["sampler"], row["steps"], row["prompt_id"]].append(row)

    design = set(itertools.product(SAMPLERS, STEPS, PROMPTS))
    if set(images) != design or set(distances) != design:
        raise ValueError("The records do not cover the 15 conditions and ten prompts.")
    if len({image["run_id"] for image in clip}) != 450 or len(clip) != 450:
        raise ValueError("Expected 450 unique measured images.")
    seed_pairs = set(itertools.combinations(GENERATION_SEEDS, 2))
    per_prompt = []
    for sampler, steps, prompt in sorted(design):
        group = images[sampler, steps, prompt]
        pair_group = distances[sampler, steps, prompt]
        run_ids = {image["seed"]: image["run_id"] for image in group}
        if len(group) != 3 or set(run_ids) != set(GENERATION_SEEDS):
            raise ValueError(f"Incomplete image seeds for {sampler}, {steps}, {prompt}.")
        if len(pair_group) != 3 or {(r["seed_a"], r["seed_b"]) for r in pair_group} != seed_pairs:
            raise ValueError(f"Incomplete seed pairs for {sampler}, {steps}, {prompt}.")
        for pair in pair_group:
            if run_ids[pair["seed_a"]] != pair["run_id_a"] or run_ids[pair["seed_b"]] != pair["run_id_b"]:
                raise ValueError("An LPIPS pair refers to an unexpected image.")
        scores = [r["clip_score"] for r in group]
        lpips = [r["lpips"] for r in pair_group]
        if not all(0 <= x <= 100 for x in scores) or not all(math.isfinite(x) and x >= -1e-6 for x in lpips):
            raise ValueError("An input score is outside the expected range.")
        per_prompt.append(dict(sampler=sampler, steps=steps, prompt_id=prompt,
                               clip_mean=statistics.mean(scores), lpips_mean=statistics.mean(lpips)))

    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(PROMPTS), size=(N_BOOTSTRAPS, len(PROMPTS)))

    def interval(values):
        means = np.asarray(values)[draws].mean(axis=1)
        return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))

    arrays = {}
    conditions = []
    for sampler, step in itertools.product(SAMPLERS, STEPS):
        group = [r for r in per_prompt if r["sampler"] == sampler and r["steps"] == step]
        if tuple(r["prompt_id"] for r in group) != PROMPTS:
            raise ValueError("Prompt ordering differs between conditions.")
        runs = [r for prompt in PROMPTS for r in images[sampler, step, prompt]]
        row = dict(sampler=sampler, steps=step, images=len(runs), prompts=len(group), lpips_pairs=30)
        for metric in ("clip", "lpips"):
            values = np.array([r[metric + "_mean"] for r in group])
            arrays[sampler, step, metric] = values
            low, high = interval(values)
            row.update({metric + "_mean": float(values.mean()), metric + "_prompt_sd": float(values.std(ddof=1)),
                        metric + "_ci95_low": low, metric + "_ci95_high": high})
        row.update(time_median_s=statistics.median(r["generation_time_s"] for r in runs),
                   peak_allocated_gib=max(r["peak_allocated_gib"] for r in runs),
                   peak_reserved_gib=max(r["peak_reserved_gib"] for r in runs))
        conditions.append(row)

    def paired_result(delta):
        low, high = interval(delta)
        return dict(clip_mean_difference=float(delta.mean()), paired_prompt_ci95_low=low,
                    paired_prompt_ci95_high=high, prompts_improved=int((delta > 0).sum()),
                    prompts_declined=int((delta < 0).sum()))

    step_comparisons = []
    for sampler in SAMPLERS:
        for before, after in STEP_COMPARISONS:
            delta = arrays[sampler, after, "clip"] - arrays[sampler, before, "clip"]
            step_comparisons.append(dict(sampler=sampler, from_steps=before, to_steps=after, **paired_result(delta)))
    sampler_comparisons = []
    for reference in ("ddim", "euler"):
        delta = arrays["dpmpp_2m", 5, "clip"] - arrays[reference, 5, "clip"]
        sampler_comparisons.append(dict(steps=5, sampler="dpmpp_2m", reference_sampler=reference, **paired_result(delta)))
    return Results(per_prompt, conditions, step_comparisons, sampler_comparisons)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_release(data):
    runs = read_csv(data / "runs.csv")
    scores = read_csv(data / "clip_scores.csv")
    by_id = {r["run_id"]: r for r in scores}
    if len(by_id) != len(scores) or {r["run_id"] for r in runs} != set(by_id):
        raise ValueError("The CLIP table and the measured run log have different image IDs.")
    images = []
    for r in runs:
        score = by_id[r["run_id"]]
        if r["phase"] != "measurement" or r["status"] != "success" or score["image_sha256"] != r["image_sha256"]:
            raise ValueError(f"Unexpected run or image digest for {r['run_id']}.")
        images.append(dict(run_id=r["run_id"], sampler=r["sampler"], steps=int(r["steps"]),
                           prompt_id=r["prompt_id"], seed=int(r["seed"]), clip_score=float(score["clip_score"]),
                           generation_time_s=float(r["generation_time_s"]), peak_allocated_gib=float(r["peak_allocated_gib"]),
                           peak_reserved_gib=float(r["peak_reserved_gib"])))
    pairs = read_csv(data / "lpips_pairs.csv")
    for row in pairs:
        for key in ("steps", "seed_a", "seed_b"):
            row[key] = int(row[key])
        row["lpips"] = float(row["lpips"])
    return images, pairs


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "results")
    args = parser.parse_args()
    results = summarize(*load_release(args.data))
    args.output.mkdir(parents=True, exist_ok=True)
    medians = {(r["sampler"], r["steps"]): r["time_median_s"] for r in results.conditions}
    conditions = [dict(r, rate_images_per_s=1 / r["time_median_s"]) for r in results.conditions]
    step_rows = [dict(r, time_ratio=medians[r["sampler"], r["to_steps"]] / medians[r["sampler"], r["from_steps"]])
                 for r in results.step_comparisons]
    save_csv(args.output / "per_prompt_metrics.csv", results.per_prompt)
    save_csv(args.output / "metrics_summary.csv", conditions)
    save_csv(args.output / "paired_step_comparisons.csv", step_rows)
    save_csv(args.output / "five_step_sampler_comparisons.csv", results.sampler_comparisons)
    print(f"Reproduced 15 conditions and two five-step sampler comparisons in {args.output}.")


if __name__ == "__main__":
    main()

