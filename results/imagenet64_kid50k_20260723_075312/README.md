# ImageNet64 50k KID Refresh

This directory contains the 50,000-sample KID refresh for the original
ImageNet64 linear, cosine, geometric-linear, and geometric-cosine slate.
Slurm array job `171748` completed all 12 schedule/objective rows with exit
code 0.

## Protocol

- Feature extractor: Inception-v3 pool3.
- Feature pool: 50,000 real and 50,000 generated images per row.
- Estimator: 100 random subsets of 1,000 real and 1,000 generated features.
- Generated samples and matching FIDs come from
  `results/imagenet64_fid50k_finished12_20260708_112342.tsv`.

## Training Provenance

These are evaluations of the original checkpoints; evaluation does not change
the dropout used during training.

| Schedule | Training dropout |
|---|---:|
| linear | 0.1 |
| cosine | 0.3 |
| geometric_linear | 0.3 |
| geometric_cosine | 0.3 |

The separately trained geometric-linear `dropout=0.1` hybrid and VLB results
are recorded under `results/geometric_linear_dropout01_richer_metrics_20260721_102718`.
