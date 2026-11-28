# Impact of Sampling Parameters on the Text–Image Alignment and Speed of Diffusion-Based Image Generation

**Alexander Stetsenko and Gil Maksim**

This repository contains the measured generation runs, image-level CLIP scores, pairwise LPIPS distances, and analysis code for the paper. Stable Diffusion 1.5 generated 450 images on a GeForce RTX 3050 Laptop GPU with 4 GiB of graphics memory and model-level CPU offloading. We compared DDIM, Euler, and DPM-Solver++ 2M at 5, 10, 20, 30, and 50 steps using the same ten prompts and three seeds. Each of the 15 configurations contributed 30 measured images. A separate warm-up for each configuration was excluded.

## Reproduce the results

From the repository directory, run the following commands in a Python environment. The analysis was verified with Python 3.12 and NumPy 2.3.5. The calculation uses the saved measurements and runs on a CPU.

```sh
python -m pip install -r requirements.txt
python sampling_statistics.py
```

The script regenerates the four CSV files in `results/`. To write them to another directory, use `python sampling_statistics.py --output reproduced`.

## Data

| File | Contents |
|---|---|
| `data/runs.csv` | One row for each of the 450 measured runs, including the sampler, step count, prompt, seed, pipeline time, GPU memory peaks, U-Net call count, safety flag, and image SHA-256 digest. |
| `data/clip_scores.csv` | One CLIP Score per image, keyed by run ID and image digest. Scores use 100 times the image–text cosine similarity, clipped below at zero. |
| `data/lpips_pairs.csv` | The three LPIPS distances between the three seeds for each prompt and configuration, with both run IDs for each pair. |

The exact timestep sequences, noise scales, sampler presets, and fixed generation settings are in `data/sampling_schedules.json`. Each sequence was checked against every recorded run for its sampler and step count.

The measurements are for 512 × 512 images generated with a guidance scale of 7.5. Generation time covers the pipeline call, including model transfers and the safety check. Image encoding and metric computation took place after timing.

## Mapping the outputs to the paper

| Output | Corresponding results |
|---|---|
| `results/metrics_summary.csv` | Tables I–III. `time_median_s` is the median of the 30 runs and `rate_images_per_s` is its reciprocal. Each memory column gives the maximum across those runs. |
| `results/per_prompt_metrics.csv` | CLIP Score averaged over three images and LPIPS averaged over three seed pairs for each of the 150 prompt–configuration combinations. |
| `results/paired_step_comparisons.csv` | Table IV uses the rows from 20 to 50 steps. `clip_mean_difference` is the later score minus the earlier score. `prompts_improved` is the number of positive prompt-level differences out of ten. `time_ratio` divides the later median time by the earlier median time. |
| `results/five_step_sampler_comparisons.csv` | The two comparisons in Section VII. Positive differences favor DPM-Solver++ 2M over the sampler named in `reference_sampler`. |

The CLIP and LPIPS condition means give each prompt equal weight. All reported 95% intervals use the 2.5th and 97.5th percentiles of 10,000 bootstrap draws of ten prompts with replacement, using seed 20261117. Each draw retains all three seeds for the sampled prompt. For paired contrasts the script resamples the within-prompt differences using the same draws for both configurations. Image-generation seeds were 42, 1234, and 2026.

At five steps, the mean paired CLIP Score difference was 0.963 points for DPM-Solver++ 2M versus DDIM (95% interval, −0.245 to 2.222) and 1.008 points versus Euler (0.209 to 2.014). These comparisons include differences in both the numerical update and the timestep schedule.
## Generation prompts

We used the same ten English prompts for all 15 combinations of sampler and denoising step count. The IDs below match the `prompt_id` values in `runs.csv`.

| ID | Prompt |
| --- | --- |
| P01 | a red car on a snowy mountain road, realistic photograph, natural lighting, detailed |
| P02 | snow-covered mountain peaks under a starry night sky, landscape photograph |
| P03 | a golden retriever sitting in a green meadow, daylight, realistic photograph |
| P04 | a blue ceramic mug on a wooden table beside a window, product photograph |
| P05 | a small stone house beside a lake, distant mountains, morning light, photograph |
| P06 | a modern living room with a grey sofa, wooden floor and large windows, interior photograph |
| P07 | portrait photograph of an adult woman wearing a green coat, plain background, soft light |
| P08 | a red apple and a yellow banana on a white plate, still life photograph |
| P09 | a yellow bicycle leaning against a blue wall, a flower pot on the ground beside it, photograph |
| P10 | a watercolor illustration of a lighthouse on a rocky island at sunset |
**### Added latency and 95% confidence intervals (Section IV.B)

The table shows how much time each screening method added on the 120 held-out
questions in each language. For each question, we subtracted the control
request time from the request time with screening enabled. The reported
estimate is the median of those 120 paired differences. All times are in
milliseconds.

| Screening method | Control | English median (95% CI), ms | Russian median (95% CI), ms |
|---|---|---:|---:|
| Detector | Baseline | 54.1 (43.8–66.2) | 101.7 (73.1–125.2) |
| Verifier | Baseline | 215.2 (200.9–236.0) | 223.0 (199.2–258.9) |
| Detector with prompt hardening | Prompt hardening alone | 30.5 (23.2–43.2) | 80.5 (25.3–130.1) |

For each estimate, we sampled 120 question IDs with replacement 10,000 times
using seed 42. We calculated the median paired difference in each sample
and used the 2.5th and 97.5th percentiles as the confidence limits.
These intervals describe variation across questions in the archived run.
Timings on another computer may differ.

To reproduce the table from the archived log, run
`python experiment/recalculate_latency.py` from the repository root.
The [archived output](experiment/latency_results.txt) also lists median
end-to-end times separately for allowed and blocked requests wherever
that group contained at least one request.**
