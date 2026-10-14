# Soperator PoC on 1-GPU H100 workers — Findings & Observations

Running log of the issues found while adapting the standard Soperator Terraform recipe to the PoC capacity, what caused them, and how they were fixed. Updated as deployment progresses.

## Context

| Item | Value |
|---|---|
| Soperator release | `soperator-v4.1.12-1` (latest stable tag; chart `4.1.12`) |
| Workers | 2 × `gpu-h100-sxm` / `1gpu-16vcpu-200gb` (1 GPU, 16 vCPU, 200 GiB each) |
| Interconnect | No InfiniBand / no GPU cluster (not supported by the 1-GPU preset) |
| Shared filesystem | 1 TiB `NETWORK_SSD` jail filestore |
| Network disk | 1 TB SSD |
| Region / project | `eu-north1` / assigned candidate project, S3 remote state |

**Core problem:** the stock recipe assumes 8-GPU H100 workers inside an InfiniBand GPU cluster. Several parts of the code hard-wire that assumption, so they had to be adapted beyond the changes listed in the setup package.

## Summary

| # | Finding | Area | Severity | Status |
|---|---|---|---|---|
| 1 | GPU fabric validation blocks GPU workers without InfiniBand | `modules/k8s` | Blocker (plan fails) | ✅ Fixed |
| 2 | `nodes_per_nodegroup` cannot be configured for non-GB300 workers | `installations/example` | Requirement gap | ✅ Fixed |
| 3 | Slurm GRES config is hard-coded for 8 GPUs | `installations/example` → `module.slurm` | Blocker (runtime: slurmd / node invalid) | ✅ Fixed |
| 4 | Network disk usage far beyond the 1 TB allocation | `terraform.tfvars` | High (runtime: PVCs Pending) | 🟢 Not a blocker: all disks provisioned; cost item only |
| 5 | Shared memory sized for the 1600 GB node | `terraform.tfvars` | Medium | 🔶 Open |
| 6 | CPU node groups oversized for a 2-GPU PoC | `terraform.tfvars` | Low (cost / quota) | 🔶 Open |
| 7 | IB-specific leftovers (health check, topology string, NCCL) | Slurm config / training | Info | ⚠️ Topology part confirmed as a blocker, see #12 |
| 8 | Dataset renders double-encoded tool-call arguments in Qwen's template | Training data | High (silently teaches the wrong output format) | ✅ Fixed in `poc/data/clean_dataset.py` |
| 9 | Disabling `nfs_in_k8s` crashes `terraform plan` | `installations/example/main.tf` | Latent (only if NFS is disabled) | ⏸ Not fixed: NFS kept enabled (stock); guard reverted 2026-10-12 |
| 10 | Race at deploy: NFS volume requested before its storage class exists | Flux release ordering | Info (self-healed in seconds) | ✅ No action |
| 11 | Console shows `ComputeInstanceStopped` warnings on every node group | mk8s node-group events | Info (normal VM boot sequence) | ✅ No action |
| 12 | GPU workers crash-loop: `worker-init` waits for InfiniBand topology labels that never exist | Soperator worker init / topology | **Blocker (Slurm workers never start)** | ✅ Fixed (kubectl labels, validated) + Terraform change ready |
| 13 | Post-deploy GPU health checks silently skip 1-GPU nodes | `soperator-activechecks` | Medium (false sense of coverage) | 📝 Documented; own validation instead |
| 14 | Dataset: ~21 % of target tool calls drop the tool's `Namespace.` prefix; ~3 % call tools not offered | Training data | High (teaches calls to non-existent function names) | ✅ Superseded by #15 (original has exact names) |
| 15 | "Cleaned" derivative dataset lost arguments and parallel calls during its conversion; the original ToolACE has correct labels | Training data (root cause of #14) | **High (~1/3 of sampled labels wrong)** | ✅ Fixed: dataset rebuilt from the original ToolACE |
| 16 | vLLM batch job never exits after finishing (background processes keep Python alive, holding the GPU) | `poc/eval/run_inference.py` | Medium (blocks scoring, holds GPU) | ✅ Fixed (explicit exit) |
| 17 | Training failed at start-up: `warmup_ratio` removed in transformers 5 | `poc/training/train.py` | Blocker (job 39) | ✅ Fixed (`warmup_steps=0.03`, versions pinned) |
| 18 | W&B showed the successful 7B run as failed: unclean shutdown after training | `poc/training/train.py` | Low (cosmetic; job COMPLETED 0:0, adapter saved) | ✅ Fixed for next runs |
| 19 | vLLM server failed at start-up: `ninja` not found (venv not activated in `serve.sbatch`) | `poc/eval/serve.sbatch` | Blocker for Exercise 2 (job 50) | ✅ Fixed |
| 20 | Live comparison got HTTP 400 from the vLLM server for multi-turn items | `poc/eval/chat_compare.py` | Medium (Exercise 2 demo client) | ✅ Fixed |

---

## Finding 1 — GPU fabric validation rejects GPU workers without InfiniBand

**Problem statement**
After removing `gpu_cluster` from the worker nodeset (as the setup package requires), `terraform plan` failed with a precondition error:

> Worker 'worker' uses GPU preset '1gpu-16vcpu-200gb' and requires either gpu_cluster.id for an existing GPU cluster or gpu_cluster.infiniband_fabric to create one.

**Root cause**
[modules/k8s/gpu_fabric_validation.tf](modules/k8s/gpu_fabric_validation.tf) treats *every* GPU worker as needing a GPU cluster:

```hcl
condition = each.value.is_gpu
  ? (each.value.has_gpu_cluster && (cluster_id != "" || fabric != ""))
  : !each.value.has_gpu_cluster
```

The 1-GPU H100 preset does not support GPU clustering, so the module's own validation blocks the only valid configuration.

**Fix** (revised 2026-10-12)
The module's preset catalogue ([modules/available_resources/preset.tf](modules/available_resources/preset.tf)) already marks every preset with `gpu_cluster_compatible` (`false` for all 1-GPU presets, `true` for 4- and 8-GPU). The node-group code already reads it ([k8s_ng_workers_v2.tf](modules/k8s/k8s_ng_workers_v2.tf) attaches a `gpu_cluster` only to compatible presets); only this precondition ignored it. The check now reads the same flag:

```hcl
gpu_cluster_compatible = module.resources.by_platform[platform][preset].gpu_cluster_compatible

condition = each.value.is_gpu && each.value.gpu_cluster_compatible
  ? (each.value.has_gpu_cluster && (length(cluster_id) > 0 || length(fabric) > 0))
  : !each.value.has_gpu_cluster
```

The error message branches the same way, so a `gpu_cluster` on a 1-GPU worker now reports "cannot join a GPU cluster, so gpu_cluster must be unset" instead of asking for one.

**Why not hard-code the preset** (first version): `preset == "1gpu-16vcpu-200gb"` missed the other 1-GPU presets (`1gpu-20vcpu-224gb`, `1gpu-24vcpu-346gb`) and gave the wrong error message. With the flag, the catalogue is the single place to describe a preset; no code change for new presets.

**Validation**
`terraform validate` passes (it cannot catch this bug: preconditions are evaluated only at plan time, with real variable values). `terraform plan`: `check_worker_gpu_fabric["0"]` and `["1"]` pass, and no `nebius_compute_v1_gpu_cluster` resource appears in the plan.

## Changes outside the setup package (quick reference)

| File | Change | Finding |
|---|---|---|
| [modules/k8s/gpu_fabric_validation.tf](modules/k8s/gpu_fabric_validation.tf) | Precondition reads the catalogue's `gpu_cluster_compatible`: 1-GPU presets must not have `gpu_cluster` | 1 |
| [installations/example/variables.tf](installations/example/variables.tf) | `nodes_per_nodegroup = optional(number)` on workers | 2 |
| [installations/example/main.tf](installations/example/main.tf) | `coalesce(nodeset.nodes_per_nodegroup, 100)` | 2 |
| [installations/example/main.tf](installations/example/main.tf) | Single-GPU `gres_config` override | 3 |
| [modules/k8s/locals.tf](modules/k8s/locals.tf), [modules/k8s/k8s_ng_workers_v2.tf](modules/k8s/k8s_ng_workers_v2.tf) | Ethernet topology labels on GPU workers without a GPU cluster | 12 |

## Troubleshooting playbook (for deploy / demo)

**Watch during `terraform apply` (second terminal).** Terraform's readiness waits run up to 60 minutes (Flux HelmReleases) plus 1 hour (Slurm cluster `Available`), so look directly instead of waiting for a timeout. Start about 15–20 minutes in, once the cluster and node groups exist:

```bash
kubectl get nodes -L slurm.nebius.ai/nodeset           # all node groups joined?
kubectl get pvc -A | grep -v Bound                     # disks Kubernetes could not create
kubectl get events -A --sort-by=.lastTimestamp | grep -i -E 'quota|provision|fail' | tail -20
kubectl -n soperator get pods                          # Pending / Init / CrashLoopBackOff
kubectl -n flux-system get helmreleases                # releases stuck not Ready
```

If a PVC is `Pending` because of quota: change the tfvars, then `terraform plan` / `apply`, then delete the **Pending** PVCs (`kubectl delete pvc <name> -n soperator`) so they are recreated with the new spec. Never delete **Bound** PVCs that hold data. Existing PVCs cannot be shrunk.

**General diagnosis**

```bash
# Startup order and the earliest failure
kubectl -n soperator get pods --sort-by=.status.startTime
kubectl get events -A --sort-by=.lastTimestamp | head -50
kubectl -n soperator logs <pod> -c <container> --previous

# PVC / disk quota problems
kubectl get pvc -A | grep -v Bound

# Flux / Helm releases
kubectl -n flux-system get helmreleases

# Slurm view
sinfo -N -l
scontrol show node
```

## Change log

| Date | Update |
|---|---|
| 2026-10-12 | Finding 9: NFS storage-class guard reverted (minimal change). NFS stays enabled as shipped, so the bug cannot trigger; documented as an observation with the one-line fix. Finding 1: check now reads `gpu_cluster_compatible` from the preset catalogue. |
| 2026-10-07 | Initial document: Findings 1–3 fixed and validated with plans v1→v2→current; Findings 4–7 open. |
| 2026-10-07 | **Exercise 2 served live** (job 53, worker-0:8000): base + `toolace` from one vLLM server; Findings 19 (ninja/venv) and 20 (OpenAI message format) fixed; live example `toolace-00488#7`: base misses the 2nd parallel call, fine-tuned matches the reference. |
| 2026-10-07 | **AFTER benchmark** (job 47, 2 min 43 s, exit 0): exact match **56.4 % → 68.2 %** (+11.8), right tool 87.6 → 91.1 %, correct no-call **60.3 → 78.0 %** (+17.7), malformed 0.4 → 0.4 %; 78 fixed vs 20 regressed; extra unrequested arguments 120 → 31. Details in TRADEOFFS D13. |
| 2026-10-07 | **7B LoRA training COMPLETED** (job 44, 43 min 44 s, exit 0): 592 steps / 2 epochs in 2,530 s (7.5 conv/s), 40.4 M trainable params (0.53 %). Train loss 1.54 → ~0.43; **eval_loss fell at every check, 0.536 → 0.433** (no overfitting); eval token accuracy 86.5 % → 88.5 %. Adapter `/data/checkpoints/qwen7b-lora/adapter` (155 MB). Harmless `torchrun` rendezvous-shutdown warning after saving. |
| 2026-10-07 | **7B training (job 44) GPU utilisation measured** (20 s average, `nvidia-smi` inside the job): worker-0 **96 %** (min 65 % during gradient sync), worker-1 **99 %**; memory ~40 / 80 GB (~50 %); power ~530–570 W. Requirement "> 80 % GPU utilisation" met on both GPUs. Loss 1.54 → 0.52 by step 82; eval_loss 0.536 → 0.504 (steps 50, 100). |
| 2026-10-07 | Training rehearsal (1.5B, 30 steps, job 40) **passed** after Finding 17 fix: loss 1.23→0.77, eval_loss 0.776, adapter saved. |
| 2026-10-07 | **BEFORE benchmark** (job 38, 1 min 41 s, exited cleanly so the Finding 16 fix works): base 7B exact match **56.4 %**, right tool 87.6 %, correct no-call 60.3 %, malformed 0.4 %; ~5.3k tok/s. Details in TRADEOFFS D13. |
| 2026-10-07 | First GPU eval (quick, 20 items): pipeline works (7/10 exact on calls, 10/10 text; 2.1 s generation). Finding 16: vLLM process didn't exit; fixed with explicit exit. |
| 2026-10-07 | **Dataset rebuilt from original ToolACE** (Finding 15 fixed): 10,514 conversations, 18,143 calls (4,125 parallel turns), 0 schema violations, 25/25 manual review; 9,462/526/526 + 725 ground-truth items; on cluster at `/data/datasets/toolace-clean/` (checksums verified). |
| 2026-10-07 | Finding 15: ~1/3 of sampled derivative labels miss user-given arguments; root cause is the derivative's conversion (original ToolACE correct, incl. ~4.1k parallel-call turns). Rebuild from original proposed. |
| 2026-10-07 | Clean dataset built (`poc/data/`): 10,547 → 10,224 conversations; Findings 8 and 14 fixed on the full data; splits uploaded to `/data/datasets/toolace-qwen-clean/` (checksums verified). |
| 2026-10-07 | Finding 14: dataset profiling: 20.7 % namespace-stripped tool names, 3.3 % unknown tools; normalisation + filter designed. |
| 2026-10-07 | Validation: **PyTorch DDP smoke test PASSED** (job 27, 32 s): 2 ranks on 2 workers, NET/Socket, all-reduce correct, ~0.94 GiB/s, 50 DDP steps, replicas identical. Fixed on the way: rendezvous picked `docker0` 169.254.0.1 via `hostname -I`, so it now resolves via `getent hosts`. |
| 2026-10-07 | Validation: 2-node nccl-tests all_reduce (MPI build) passed: ranks on both workers, NET/Socket over eth0, 0 errors, busbw ~1.1 GB/s. Plain `all_reduce_perf` is not MPI-linked (runs 1 rank per node); use `*_perf_mpi`. |
| 2026-10-07 | Finding 12 made permanent: label change applied in place (no node rollout). Follow-up plan shows only the 3 routine bookkeeping refreshes, so code and cluster are in sync. |
| 2026-10-07 | **`terraform apply` complete: 49 added, 0 changed, 0 destroyed** (~50 min including the Finding 12 fix). All 15 Slurm active-check jobs `COMPLETED` (`sacct`); both workers `idle`, no drain reason; `login.sh` → <login-node-ip>. |
| 2026-10-07 | Finding 13: activechecks GPU tests exit 0 (`Unsupported platform`) on 1-GPU nodes; any failed waited check would fail `terraform apply`. |
| 2026-10-07 | Finding 12 fixed: second bug found (tier-0 vs tree topology mismatch inside Soperator); `tier-1`-only label. Both workers idle in Slurm under `eth-spine`. |
| 2026-10-07 | Finding 12: GPU workers crash-loop in `worker-init` `wait-topology`: no `topology.nebius.com/tier-*` labels without InfiniBand. Root cause traced in the soperator 4.1.12 source; node-label fix proposed. |
| 2026-10-07 | populate-jail completed in 8 min (12.7 GB, 102 k files). All PVCs Bound, so quota OK for the full stock footprint. Correction: MariaDB uses its own 512 GiB disk, not the accounting filestore. |
| 2026-10-07 | Deploy: system node group autoscaled 3→4→5 for VictoriaMetrics (6 vCPU / 24 GiB request); Finding 6 updated. |
| 2026-10-07 | First apply in progress: all 7 node groups RUNNING (2 × 1-GPU H100, `nvidia.com/gpu=1` each). Finding 10: NFS StorageClass race (self-healed). 3.7 TiB NFS disk Bound, so quota OK. |
| 2026-10-07 | Finding 9: plan crashes when NFS is disabled; guard added in main.tf (no effect while NFS is enabled). Finding 4: NFS disable tested (plan v3, 46 resources) then reverted; deploying with stock storage first, CSA asked. |
| 2026-10-07 | Finding 8: dataset tool-call arguments double-encoded by Qwen's template; fix designed. |
