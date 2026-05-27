# Full Evaluation Result Snapshot

This snapshot records the full NLL/FID slate available on 2026-05-26:

- 4 datasets: `mnist`, `fashionmnist`, `cifar10`, `imagenet64`
- 3 objectives: `simple`, `hybrid`, `vlb`
- 7 schedules: `linear`, `cosine`, `geometric_linear`, `geometric_cosine`, `tuned_nll`, `tuned_balanced`, `tuned_fid`
- 84 total rows

The machine-readable table is in `evaluation_results_full.tsv`.

## Columns

- `dataset`: evaluated dataset.
- `schedule`: step-size/noise schedule family.
- `objective`: training objective.
- `beta_1`, `alpha_bar_T`: geometric endpoint parameters for the tuned geometric schedules. These are blank for linear, cosine, and matched geometric baselines.
- `nll_bpd`: NLL in bits per dimension.
- `fid`: FID score.
- `nll_samples`: number of test/validation images used for NLL evaluation.
- `fid_samples`: number of generated images used for FID.
- `sampling_steps`: denoising steps used during sampling. These results used full 4000-step sampling, not a respaced sampler.
- `source`: original evaluation directory or aggregate TSV used to recover the number.

## Tuned Schedule Parameters

| Schedule | `beta_1` | `alpha_bar_T` |
|---|---:|---:|
| `tuned_nll` | `1e-5` | `3e-3` |
| `tuned_balanced` | `3e-5` | `1e-3` |
| `tuned_fid` | `3e-3` | `1e-2` |

## Best Rows

| Dataset | Best NLL | Best FID |
|---|---|---|
| `mnist` | `geometric_cosine/hybrid`: 0.286482 | `cosine/vlb`: 0.432768 |
| `fashionmnist` | `geometric_cosine/hybrid`: 0.761154 | `geometric_cosine/hybrid`: 1.230924 |
| `cifar10` | `tuned_nll/vlb`: 2.919428 | `tuned_fid/simple`: 5.042053 |
| `imagenet64` | `tuned_nll/hybrid`: 3.303356 | `cosine/simple`: 27.780287 |

## Notes

- Lower is better for both `nll_bpd` and `fid`.
- MNIST, FashionMNIST, and CIFAR10 FID rows use 50,000 generated samples.
- ImageNet64 FID rows use 10,000 generated samples.
- The original result files remain on project storage; this repo-local snapshot preserves the numeric results and source paths for future reference.
