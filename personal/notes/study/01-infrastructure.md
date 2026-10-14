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

<a id="challenge-1"></a>

### 1. Plan blocked: GPU workers "must" have a GPU cluster
> **Changed:** `gpu_fabric_validation.tf`: precondition now reads the preset catalogue's `gpu_cluster_compatible` flag, so 1-GPU presets must *not* have a `gpu_cluster`.

- Found: `terraform plan` precondition failed in `check_worker_gpu_fabric` (`terraform validate` passes: preconditions run only at plan time).
- Cause: the check requires `gpu_cluster` for every GPU preset, ignoring the catalogue's `gpu_cluster_compatible = false` for 1-GPU presets, which the node-group code already respects.
- References:
  - Preset catalogue, `gpu_cluster_compatible` per GPU preset: [`modules/available_resources/preset.tf` L227–L325](../modules/available_resources/preset.tf#L227-L325) (H100 `1gpu-16vcpu-200gb`: [L228–L232](../modules/available_resources/preset.tf#L228-L232))
  - Node-group code already using the flag: [`modules/k8s/k8s_ng_workers_v2.tf` L150](../modules/k8s/k8s_ng_workers_v2.tf#L150)
  - The fixed check: [`modules/k8s/gpu_fabric_validation.tf` L13–L34](../modules/k8s/gpu_fabric_validation.tf#L13-L34)
- Fix: `modules/k8s/gpu_fabric_validation.tf`: require a `gpu_cluster` only when `is_gpu && gpu_cluster_compatible`; otherwise it must be unset, with a matching error message. Covers all 1-GPU presets, no hard-coded names; 4/8-GPU presets unchanged.

<a id="challenge-2"></a>

### 2. Slurm GPU config described 8 GPUs (`/dev/nvidia0–7`) on a 1-GPU node
> **Changed:** `main.tf`: for `gpu-h100-sxm` + `1gpu-16vcpu-200gb`, `gres_config` is one line `File=/dev/nvidia0` instead of the platform's 8-GPU list.

- Found: in the plan's rendered NodeSet, `Gres=gpu:…:1` but 8 `gresConfig` lines with `Cores=0-63`.
- Cause: `gres.tf` keys GRES by **platform**, not preset, so every H100 preset gets the 8-GPU layout.
- Fix: `installations/example/main.tf`: single line `File=/dev/nvidia0` for the 1-GPU preset. Verified with `scontrol show node` → `Gres=…:1`, `CPUTot=16`.

<a id="challenge-3"></a>

### 3. Couldn't split the 2 GPUs into separate node groups
> **Changed:** `variables.tf`: added `nodes_per_nodegroup = optional(number)` to workers; `main.tf`: `coalesce(nodeset.nodes_per_nodegroup, 100)`.

- Found: the plan created 1 node group with 2 nodes; `nodes_per_nodegroup` was rejected / ignored.
- Cause: not in the worker variable schema; non-GB300 workers hard-coded to 100 per group.
- Fix: `variables.tf` adds optional `nodes_per_nodegroup`; `main.tf` uses `coalesce(…, 100)`. Result: `worker-0`, `worker-1` × 1 node; zero-surge updates never need a 3rd GPU.

<a id="challenge-4"></a>

### 4. GPU workers crash-looped: Soperator waited for InfiniBand topology labels
> **Changed:** `locals.tf` + `k8s_ng_workers_v2.tf`: GPU workers without a `gpu_cluster` get node label `topology.nebius.com/tier-1=eth-spine`.

- Found: worker pods `Init:CrashLoopBackOff`; `worker-init` log: `Topology … not found after 180s; Available nodes in ConfigMap: []`.
- Traced in the Soperator source:
  - `wait-topology` runs whenever a topology plugin is set (always `topology/tree`).
  - It needs `topology.nebius.com/tier-*` node labels, which Nebius only sets on IB GPU-cluster nodes.
- Second bug while fixing it: with `tier-0` + `tier-1`, `scontrol update … topology=` failed ("Requested topology configuration is not available"). Soperator's `topology.conf` builder ignores `tier-0`, but `worker_init.py` uses it.
- Fix:
  - Unblocked live with `kubectl label … topology.nebius.com/tier-1=eth-spine` (`tier-1` only).
  - Made permanent in Terraform (`modules/k8s/locals.tf`, `k8s_ng_workers_v2.tf`): the label is applied only to GPU workers **without** a `gpu_cluster`.
  - Applied in place, with no node replacement.

<a id="challenge-5"></a>

### 5. Stock `terraform.tfvars` mounted a placeholder existing filestore
> **Changed:** `terraform.tfvars`: commented out the `data` jail submount (`computefilesystem-<YOUR-FILESTORE-ID>`) and set `allow_empty_jail_submounts = true`.

- Found: the stock example mounts an *existing* filestore at `/mnt/data` by a placeholder ID; the allocation has no such filestore.
- Fix: no extra submount; the 1 TiB jail is the only shared filesystem, matching the allocation.

## Terraform changes (attached)

Changes made to the Terraform recipe to fix the challenges above. The **Issue** link jumps to the full write-up (Found / Cause / Fix).

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
