# Experiment Registry

Last updated: 2026-06-23

This file is the first place to look before searching GPFS. The companion
machine-readable index is `results/experiment_registry.tsv`.

## Primary Aggregates

| experiment group | local result file | GPFS location | coverage | notes |
|---|---|---|---|---|
| Unconditional NLL/FID aggregate | `evaluation_results_full.tsv` | exact per-run directories are in the `source` column | MNIST, FashionMNIST, CIFAR-10, ImageNet64; simple/hybrid/vlb; linear, cosine, geometric_linear, geometric_cosine, tuned_nll, tuned_fid, tuned_balanced | 84 result rows. This is the main table for the original unconditional slate plus tuned geometric schedules. |
| Baseline/geometric richer metrics | `results/richer_metrics/richer_metrics_summary.tsv` | exact per-run directories are in `results/richer_metrics/manifest.tsv` | MNIST, FashionMNIST, CIFAR-10, ImageNet64; simple/hybrid/vlb; linear, cosine, geometric_linear, geometric_cosine | CMMD, KID, Inception density/coverage, and CLIP density/coverage. Submission records: `results/submitted_jobs_20260601.tsv`, `results/clip_density_coverage_submissions_20260617_103213.tsv`. |
| Linabar NLL/FID | not yet aggregated into one local TSV | `/project_gpfs/bata0/bjin0/linabar_evaluation_nll_fid_20260605_041727` | MNIST, FashionMNIST, CIFAR-10, ImageNet64; simple/hybrid/vlb; linabar_linear, linabar_cosine | Per-run result files live under the GPFS root. The richer-metrics manifest also records sample paths. |
| Linabar richer metrics | `results/linabar_richer_metrics_20260605_041727/richer_metrics_summary.tsv` | samples come from `/project_gpfs/bata0/bjin0/linabar_evaluation_nll_fid_20260605_041727` | completed linabar rows with available samples | CMMD, KID, Inception density/coverage, and CLIP density/coverage. Submission records: `results/linabar_richer_metrics_20260605_041727/submission.tsv`, `results/clip_density_coverage_submissions_20260617_103213.tsv`. |
| SVHN NLL/FID | `results/svhn_evaluation_nll_fid_20260610_100613.tsv` | `/project_gpfs/bata0/bjin0/svhn_evaluation_nll_fid_20260610_100613` | SVHN; simple/hybrid/vlb; linear, cosine, geometric_linear, geometric_cosine, linabar_linear, selected linabar_cosine | 17 evaluated rows. `linabar_cosine_hybrid` was skipped because training produced NaNs. |
| SVHN richer metrics | `results/svhn_richer_metrics_20260615_080703/richer_metrics_summary.tsv` | samples come from `/project_gpfs/bata0/bjin0/svhn_evaluation_nll_fid_20260610_100613` | same evaluated SVHN rows | CMMD, KID, Inception density/coverage, and CLIP density/coverage. Manifest: `results/svhn_richer_metrics_20260615_080703/manifest.tsv`. |
| Toy oracle KL | GPFS summary only | `/project_gpfs/bata0/bjin0/toy_oracle_kl_20260601_072252` | 7 toy distributions; T in {50, 100, 250, 1000}; linear, cosine, geometric_linear, geometric_cosine, linabar_linear, linabar_cosine | Main files: `summary.tsv`, `summary.jsonl`, and per-distribution plots. Submission record: `results/submitted_jobs_20260601.tsv`. |
| Conditional CIFAR-10 NLL/FID | `results/conditional_cifar10_nll_fid_20260531_134458.tsv` | `/project_gpfs/bata0/bjin0/cifar10_conditional_evaluation_20260531_134458` | CIFAR-10 class-conditional; simple/hybrid/vlb; cosine, geometric_cosine | Training jobs 122553-122558 and evaluation jobs 138197-138202 all completed. |
| Conditional CIFAR-10 richer metrics | `results/conditional_cifar10_richer_metrics_20260615_152836/richer_metrics_summary.tsv` | samples come from `/project_gpfs/bata0/bjin0/cifar10_conditional_evaluation_20260531_134458` | CIFAR-10 class-conditional; simple/hybrid/vlb; cosine, geometric_cosine | CMMD, KID, Inception density/coverage, and CLIP density/coverage. Completed as Slurm array jobs 153666 and 154733. Manifest: `results/conditional_cifar10_richer_metrics_20260615_152836/manifest.tsv`. The earlier 153651 run is invalid because it reused unconditional generated-feature caches. |
| CIFAR-100 dataset prep | `/project_gpfs/bata0/bjin0/cifar100_32x32/cifar10_overlap_report.tsv` | `/project_gpfs/bata0/bjin0/cifar100_32x32` | train/test PNG folders with 50,000/10,000 images | Prep job 153770 completed. Exact pixel-hash overlap with CIFAR-10: 1 train image and 2 test images. |
| CIFAR-100 NLL/FID | `results/cifar100_evaluation_nll_fid_cifar100_evaluation_nll_fid_20260617_122440.tsv` | `/project_gpfs/bata0/bjin0/cifar100_evaluation_nll_fid_20260617_122440` | CIFAR-100; simple/hybrid/vlb; linear, cosine, geometric_linear, geometric_cosine | Completed. All 12 evaluation directories have `nll_results.txt`, `fid_results.txt`, and `samples_50000x32x32x3.npz`. Manifest: `/project_gpfs/bata0/bjin0/cifar100_evaluation_nll_fid_20260617_122440/submission.tsv`. |
| CIFAR-100 richer metrics | `results/cifar100_richer_metrics_20260623_230233/richer_metrics_summary.tsv` | samples come from `/project_gpfs/bata0/bjin0/cifar100_evaluation_nll_fid_20260617_122440` | CIFAR-100; simple/hybrid/vlb; linear, cosine, geometric_linear, geometric_cosine | Completed as Slurm array job 157934. Metrics: CMMD, KID, Inception density/coverage, and CLIP density/coverage. Manifest: `results/cifar100_richer_metrics_20260623_230233/manifest.tsv`. |
| CelebA-64 dataset prep | none | `/project_gpfs/bata0/bjin0/celeba_64x64` | train/valid/test PNG folders with 162,770/19,867/19,962 images | Prep job 156065 completed 2026-06-20 after adding `gdown` to the repo-local `.venv`. Source files are under `/project_gpfs/bata0/bjin0/celeba_64x64/source`. |
| LSUN Bedroom-64 dataset prep | none | `/project_gpfs/bata0/bjin0/lsun_bedroom_64x64` | LSUN Bedroom train/val LMDB source for 64x64 experiments | Prep job 156234 completed 2026-06-21 in 02:15:03. `_source_manifest.tsv` reports 3,033,042 train entries and 300 val entries; PNG conversion is disabled by default to conserve GPFS inodes. |
| CelebA-64 NLL/FID | pending aggregation to `results/celeba64_evaluation_nll_fid_celeba64_evaluation_nll_fid_20260620_224700.tsv` | `/project_gpfs/bata0/bjin0/celeba64_evaluation_nll_fid_20260620_224700` | CelebA-64; hybrid/vlb; linear, cosine, geometric_linear, geometric_cosine | Evaluation jobs 156155-156162 are running. All have written `nll_results.txt` and are generating FID samples. Manifest: `/project_gpfs/bata0/bjin0/celeba64_evaluation_nll_fid_20260620_224700/submission.tsv`. |

## Training Roots

| group | GPFS training root or manifest | notes |
|---|---|---|
| Linabar full slate | `/project_gpfs/bata0/bjin0/linabar_full_slate_20260528_192810` and `/project_gpfs/bata0/bjin0/linabar_resume_full_slate_20260531_131307/submission.tsv` | Resumed after earlier storage failures. |
| SVHN full slate | `/project_gpfs/bata0/bjin0/svhn_full_slate_20260604_031305/submission.tsv` | 18 training jobs; one final run was unusable for evaluation due NaNs. |
| Conditional CIFAR-10 canonical training checkpoints | `/project_gpfs/bata0/bjin0/bjin0/122553/logs` through `/project_gpfs/bata0/bjin0/bjin0/122558/logs` | These are the checkpoints used by the conditional CIFAR-10 evaluation manifest. |
| CIFAR-100 reduced slate | `/project_gpfs/bata0/bjin0/cifar100_full_slate_20260615_184504/submission.tsv` | 12 jobs completed: simple/hybrid/vlb x linear/cosine/geometric_linear/geometric_cosine. Initial jobs 153779-153782 failed but resumed as 155435-155438 and completed successfully. |
| CelebA-64 reduced slate | `/project_gpfs/bata0/bjin0/celeba64_full_slate_20260620_165200/submission.tsv` | 8 jobs completed successfully: hybrid/vlb x linear/cosine/geometric_linear/geometric_cosine. Jobs 156066-156073 used prepared dataset `/project_gpfs/bata0/bjin0/celeba_64x64/train`. Earlier dependency-blocked jobs 154839-154846 were canceled. |
| LSUN Bedroom-64 reduced slate | `/project_gpfs/bata0/bjin0/lsun_bedroom64_full_slate_20260622_101208/submission.tsv` | 8 jobs submitted 2026-06-22: hybrid/vlb x linear/cosine/geometric_linear/geometric_cosine. Jobs 156585-156592 use LMDB train source `/project_gpfs/bata0/bjin0/lsun_bedroom_64x64/source/bedroom_train_lmdb`. |

## Lookup Pattern

1. Start with `results/experiment_registry.tsv` or this file.
2. For NLL/FID, use the local TSV if present.
3. For samples and per-run logs, use the `source`, `source_eval_dir`, or `gpfs_root` recorded by the local table.
4. Only run a GPFS `find` when adding a new experiment group to this registry.
