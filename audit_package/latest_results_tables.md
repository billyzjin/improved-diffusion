# Latest Existing Evaluation Tables

Generated: April 22, 2026

Selection rule: use the latest completed evaluation summaries already present on disk. The April baseline reruns have not yet produced newer evaluation summaries, so baseline cells use the January/February completed summaries. The geometric cells use the April geometric evaluation summaries.

All metrics are lower-is-better. The best value in each row is bolded.

## NLL

NLL is reported in bits/dim.

| Dataset | Objective | linear | cosine | geometric_linear | geometric_cosine |
|---|---|---:|---:|---:|---:|
| MNIST | Simple | 0.959524 | 0.754538 | 0.647220 | **0.343540** |
| MNIST | Hybrid | 0.387824 | 0.346260 | 0.345607 | **0.286482** |
| MNIST | VLB | 0.476889 | 0.315578 | 0.391178 | **0.312752** |
| Fashion-MNIST | Simple | 1.452883 | 1.214592 | 1.746214 | **0.862070** |
| Fashion-MNIST | Hybrid | 0.947477 | 0.827938 | 0.887216 | **0.761154** |
| Fashion-MNIST | VLB | 1.096259 | 0.816530 | 0.921555 | **0.773261** |
| CIFAR-10 | Simple | 3.418211 | **3.298120** | 8.921947 | 4.874985 |
| CIFAR-10 | Hybrid | 3.295683 | 3.206944 | 3.012372 | **2.921334** |
| CIFAR-10 | VLB | 3.090245 | 2.981294 | 3.009022 | **2.924799** |
| ImageNet-64 | Simple | 3.899004 | **3.829042** | 11.811029 | 6.747508 |
| ImageNet-64 | Hybrid | 3.801615 | 3.783328 | **3.714664** | 3.722687 |
| ImageNet-64 | VLB | **3.663513** | 3.735592 | 3.666266 | 3.680610 |

## FID

| Dataset | Objective | linear | cosine | geometric_linear | geometric_cosine |
|---|---|---:|---:|---:|---:|
| MNIST | Simple | 2.598570 | 2.173214 | 0.516012 | **0.482725** |
| MNIST | Hybrid | 2.573753 | 2.258499 | 0.494281 | **0.434488** |
| MNIST | VLB | 0.579668 | **0.432768** | 0.609716 | 0.450340 |
| Fashion-MNIST | Simple | 2.435917 | 2.221193 | 1.512209 | **1.438355** |
| Fashion-MNIST | Hybrid | 2.320925 | 2.352145 | 1.546636 | **1.230924** |
| Fashion-MNIST | VLB | 1.277860 | 1.772126 | 1.399474 | **1.254050** |
| CIFAR-10 | Simple | 9.768684 | **5.549630** | 6.795889 | 9.684377 |
| CIFAR-10 | Hybrid | 8.420695 | **5.474877** | 7.114203 | 9.037849 |
| CIFAR-10 | VLB | **10.523915** | 11.756001 | 11.289727 | 10.693692 |
| ImageNet-64 | Simple | **14.385173** | 14.900750 | 24.495095 | 26.631956 |
| ImageNet-64 | Hybrid | **10.866806** | 13.164316 | 26.099496 | 25.785379 |
| ImageNet-64 | VLB | 31.413560 | **29.936082** | 40.572724 | 39.147460 |

## TV

| Dataset | Objective | linear | cosine | geometric_linear | geometric_cosine |
|---|---|---:|---:|---:|---:|
| MNIST | Simple | 0.025871 | 0.028060 | **0.021873** | 0.023232 |
| MNIST | Hybrid | 0.029066 | 0.029549 | **0.017830** | 0.019255 |
| MNIST | VLB | **0.008747** | 0.013407 | 0.010124 | 0.013827 |
| Fashion-MNIST | Simple | **0.028004** | 0.037592 | 0.036994 | 0.037305 |
| Fashion-MNIST | Hybrid | **0.028943** | 0.035869 | 0.030763 | 0.034868 |
| Fashion-MNIST | VLB | **0.020381** | 0.038582 | 0.033098 | 0.032345 |
| CIFAR-10 | Simple | 0.024988 | 0.016337 | 0.017036 | **0.015082** |
| CIFAR-10 | Hybrid | 0.027039 | **0.017260** | 0.021700 | 0.018926 |
| CIFAR-10 | VLB | 0.015822 | 0.015399 | 0.021130 | **0.014506** |
| ImageNet-64 | Simple | **0.033862** | 0.056038 | 0.055604 | 0.077208 |
| ImageNet-64 | Hybrid | **0.037480** | 0.048143 | 0.078703 | 0.058296 |
| ImageNet-64 | VLB | 0.157666 | **0.102715** | 0.297331 | 0.266568 |

## Sources

Baseline summaries:

| Dataset | Source |
|---|---|
| MNIST | `/project_gpfs/bata0/bjin0/mnist_evaluation_parallel_20260128_083113/results_summary.txt` |
| Fashion-MNIST | `/project_gpfs/bata0/bjin0/fashion_evaluation_combined_latest_20260127_174224/results_summary.txt` |
| CIFAR-10 | `/project_gpfs/bata0/bjin0/cifar10_evaluation_combined_latest_20260127_174106/results_summary.txt` |
| ImageNet-64 | `/project_gpfs/bata0/bjin0/imagenet64_evaluation_parallel_20260127_123517/results_summary.txt` |

Geometric summaries:

| Dataset | Source |
|---|---|
| MNIST | `/project_gpfs/bata0/bjin0/mnist_evaluation_parallel_geometric_20260411_090347/results_summary.txt` |
| Fashion-MNIST | `/project_gpfs/bata0/bjin0/fashionmnist_evaluation_parallel_geometric_20260410_231225/results_summary.txt` |
| CIFAR-10 | `/project_gpfs/bata0/bjin0/evaluation_parallel_geometric_20260410_231224/results_summary.txt` |
| ImageNet-64 | `/project_gpfs/bata0/bjin0/imagenet64_evaluation_parallel_geometric_20260410_231226/results_summary.txt` |
