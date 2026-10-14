# 1. Infrastructure: Soperator on Nebius (2 × 1-GPU H100, no InfiniBand)

## Achieved

- Deployed **Soperator `soperator-v4.1.12-1`** (latest stable tag) with Terraform on Nebius Managed Kubernetes (`eu-north1`), using the provided S3 remote state.
- **2 GPU workers**, `gpu-h100-sxm` / `1gpu-16vcpu-200gb`, as **2 node groups × 1 node**, `max_surge = 0`, on the reserved H100 allowance. No GPU cluster / InfiniBand.
- 1 TiB shared jail filesystem; Slurm cluster `Available`; both workers `idle` with `Gres=gpu:nvidia_h100_80gb_hbm3:1`.
- **Code and cluster in sync:** after the final apply, `terraform plan` shows **0 to change** (only the 3 resources the recipe always re-runs).
- **Validated end to end:**
  - 2-node NCCL `all_reduce_perf_mpi`: NET/Socket over `eth0`, **0 errors, ~1.1 GB/s**
  - PyTorch DDP smoke test: `SMOKE TEST PASSED`
  - 7B LoRA DDP training: **96–100 % GPU utilisation on both nodes** (requirement: > 80 %)

NCCL test (run from the login node):

```bash
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
srun -N2 --ntasks-per-node=1 --gpus-per-node=1 --mpi=pmix \
  all_reduce_perf_mpi -b 8 -e 1G -f 2 2>&1 | tee ~/nccl-2node.log
```

- `NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME=eth0`: no InfiniBand, so NCCL uses TCP sockets on `eth0` (not `docker0`).
- `all_reduce_perf_mpi` with `--mpi=pmix`: one 2-rank test across both nodes.
- `-b 8 -e 1G -f 2`: message sizes from 8 B to 1 GiB, doubling each step. Results are checked for correctness by default.

## Challenges and fixes

| # | Challenge | Solution |
|---|---|---|
| <a id="challenge-1"></a>1 | **Terraform rejected the GPU workers.** The GPU fabric validation (a plan-time precondition) required every GPU worker to have a `gpu_cluster` (InfiniBand fabric). 1-GPU H100 presets can't join a GPU cluster, so `terraform plan` failed. | The check now reads the preset catalogue's `gpu_cluster_compatible` flag. A `gpu_cluster` is required only for 4- and 8-GPU presets and must be unset for 1-GPU presets. No preset names are hard-coded. |
| <a id="challenge-2"></a>2 | **Slurm GRES config described 8 GPUs on a 1-GPU node.** GRES is keyed by platform, not preset, so every `gpu-h100-sxm` node got the 8-GPU layout (`/dev/nvidia0–7`). | 1-GPU H100 nodes now get a single-GPU `gres_config` (`File=/dev/nvidia0`). `scontrol show node` shows `Gres=gpu:nvidia_h100_80gb_hbm3:1` and 16 CPUs. |
| <a id="challenge-3"></a>3 | **Both GPUs landed in one node group.** The worker schema had no `nodes_per_nodegroup` option, and non-GB300 workers were hard-coded to 100 nodes per group. | Added an optional `nodes_per_nodegroup` setting, defaulting to 100, and set it to 1. Each GPU node now has its own node group, so updates with `max_surge = 0` never need a 3rd GPU. |
| <a id="challenge-4"></a>4 | **GPU workers crash-looped (`Init:CrashLoopBackOff`).** Soperator's `worker-init` waits for `topology.nebius.com/tier-*` node labels whenever the `topology/tree` plugin is set. Nebius adds those labels only on InfiniBand GPU-cluster nodes. A second bug: with a `tier-0` label, `scontrol update … topology=` failed, because Soperator's `topology.conf` builder ignores `tier-0` but `worker_init.py` uses it. | Unblocked the cluster live with `kubectl label … topology.nebius.com/tier-1=eth-spine`, with `tier-1` only. Then made the label permanent in Terraform for GPU workers without a `gpu_cluster`. It was applied in place, with no node replacement. |
| <a id="challenge-5"></a>5 | **The example config mounted a placeholder filestore.** The stock example mounts an existing filestore at `/mnt/data` by a placeholder ID, and this project has none. | Removed that jail submount and set `allow_empty_jail_submounts = true`, so the new 1 TiB jail filesystem is the only shared filesystem. |

## Terraform changes (attached)

Changes made to the Terraform recipe to fix the challenges above. The **Issue** column links to the matching challenge.

| File | Change | Issue |
|---|---|---|
| [`modules/k8s/gpu_fabric_validation.tf`](../modules/k8s/gpu_fabric_validation.tf) | Precondition reads `gpu_cluster_compatible`: 1-GPU presets accepted without a GPU cluster | [Challenge 1](#challenge-1) |
| [`installations/example/main.tf`](../installations/example/main.tf) | Single-GPU GRES `File=/dev/nvidia0` instead of the 8-GPU layout | [Challenge 2](#challenge-2) |
| [`installations/example/variables.tf`](../installations/example/variables.tf), [`main.tf`](../installations/example/main.tf) | `nodes_per_nodegroup` support, so the value from the instructions takes effect | [Challenge 3](#challenge-3) |
| [`modules/k8s/locals.tf`](../modules/k8s/locals.tf), [`modules/k8s/k8s_ng_workers_v2.tf`](../modules/k8s/k8s_ng_workers_v2.tf) | `topology.nebius.com/tier-1` label for GPU workers without a GPU cluster | [Challenge 4](#challenge-4) |
| [`installations/example/terraform.tfvars`](../installations/example/terraform.tfvars) | Removed the stock placeholder `data` jail submount (an existing filestore ID) and set `allow_empty_jail_submounts = true`, so the 1 TiB jail is the only shared filesystem, matching the allocation | [Challenge 5](#challenge-5) |

**Set as per the setup instructions:**
- `public_o11y_enabled = false`
- `backups_enabled = "force_disable"`
- Workers: 2 × `gpu-h100-sxm` / `1gpu-16vcpu-200gb`, `nodes_per_nodegroup = 1`, no `gpu_cluster`
- New 1 TiB jail filestore
- S3 remote state with the provided backend credentials

**Masked in the zip:** only the S3 backend keys `TF_BACKEND_AWS_ACCESS_KEY_ID` and `TF_BACKEND_AWS_SECRET_ACCESS_KEY` in `installations/example/soperator-ganguly-sandip-backend-secret.env`. Replace them with the real values (and optionally the SSH key in `terraform.tfvars`) before `terraform apply`; see [03-reproduce.md](03-reproduce.md).
