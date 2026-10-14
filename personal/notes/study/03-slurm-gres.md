# 3. Slurm GPU requests: `--gres` and sbatch vs srun

## `--gres=gpu:nvidia_h100_80gb_hbm3:1`

"Give the job 1 NVIDIA H100 80GB GPU on each node it runs on."

```
--gres = gpu : nvidia_h100_80gb_hbm3 : 1
  │       │            │               │
  │       │            │               └── count: 1 GPU PER NODE
  │       │            └── type: GPU model
  │       └── resource name: GPUs
  └── "generic resource"
```

| Part | Meaning |
|---|---|
| `--gres` | **Gen**eric **Res**ource. CPUs/memory are native; GPUs, NICs, licences are "generic". |
| `gpu` | Resource name |
| `nvidia_h100_80gb_hbm3` | GPU type label. H100, 80 GB, HBM3 memory. **Not autodetected:** our Terraform hard-codes it (see "How Slurm knows the node has a GPU" below). |
| `1` | Count **per node**, not total |

**Ties to memory math:** 80 GB is the budget. LoRA 7B (~15 GB) fits on `:1`. Full fine-tune (~112 GB) doesn't.

## Variations

| You write | Meaning |
|---|---|
| `--gres=gpu:1` | 1 GPU of any type |
| `--gres=gpu:nvidia_h100_80gb_hbm3:8` | All 8 H100s on each node |
| `--nodes=2 --gres=gpu:...:8` | 2 × 8 = 16 GPUs total |
| `--gpus=16` | 16 GPUs total, Slurm spreads them |
| `--gpus-per-node=8` | Newer way to write `--gres=gpu:8` |

Check the type string on the cluster:
```bash
sinfo -o "%N %G"
scontrol show node <node> | grep -i gres
```

A wrong type string → job stays pending (`Invalid generic resource` or reason `Resources`). Plain `gpu:1` avoids that on a single-GPU-type cluster.

## Where do you put it? sbatch vs srun

Works with **`sbatch`, `srun` and `salloc`**, at different levels:

- `sbatch` / `salloc` **reserve** resources for the whole job (the allocation).
- `srun` **runs a step** inside the reservation, using all or part of it.

**Four places:**

1. **`#SBATCH` line in the script** (most common). Our `train.sbatch`:
   ```bash
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=1
   #SBATCH --gpus-per-node=1        # same as --gres=gpu:1
   ```
2. **sbatch command line** (overrides `#SBATCH`): `sbatch --gpus-per-node=8 train.sbatch`
3. **srun inside the script:** not needed. A step gets **all** the job's GPUs by default. Only add it for a subset.
4. **srun on its own** (reserve + run in one go): `srun -N1 --gres=gpu:1 nvidia-smi`

**Precedence:** `sbatch` command line > `SBATCH_*` env vars > `#SBATCH` lines.

```
sbatch  ──►  "Reserve 2 nodes × 1 H100"     ← GPU request lives here
   │
   └── srun torchrun ...                    ← step uses what was reserved
          ├── node 1 → GPU 0
          └── node 2 → GPU 0
```

`srun` can never get more than `sbatch` reserved.

**Pattern:** GPU request in the `#SBATCH` header, none on `srun`.

## How Slurm knows the node has a GPU

**Short answer:** our code declares it, and the hardware only has to back it up. With `AutoDetect=off`, Slurm does not ask the GPU what it is.

**Analogy:** a hotel room listing. Terraform writes the listing ("Room 1: one H100 bed"). slurmd checks the bed is physically there (`/dev/nvidia0` exists) before opening the room. Slurm never checks the brand of the bed.

### The chain

```
1. Terraform: modules/available_resources/
     gres.tf     platform gpu-h100-sxm  →  type "nvidia_h100_80gb_hbm3"
     preset.tf   preset 1gpu-16vcpu-200gb → gpus = 1
          │
2. Terraform: installations/example/main.tf (~line 566)
     gres_name   = type string
     gres_config = our 1-GPU override (FINDINGS #3)
          │
3. Template: modules/slurm/templates/helm_values/flux_release_nodesets.yaml.tftpl
     static:     "CPUs=16 ... Gres=gpu:nvidia_h100_80gb_hbm3:1"   → the node's line in slurm.conf
     gresConfig: "AutoDetect=off Name=gpu Type=... File=/dev/nvidia0 Flags=nvidia_gpu_env"   → gres.conf
          │
4. Soperator operator renders slurm.conf + gres.conf into a ConfigMap "<cluster>-slurm-configs"
   → sconfigcontroller copies them into the shared jail: /etc/slurm/  (see "See the real files" below)
          │
5. slurmd starts on the worker → reads gres.conf → checks /dev/nvidia0 exists
     → registers with slurmctld: "1 × gpu:nvidia_h100_80gb_hbm3"
          │
6. slurmctld compares with slurm.conf
     match    → node IDLE, schedulable
     mismatch → node INVALID / DRAIN
```

### Two files, two jobs

| File | Says | Role |
|---|---|---|
| `slurm.conf` (`static:` line) | `Gres=gpu:nvidia_h100_80gb_hbm3:1` | **What the node should have** (the scheduler's view) |
| `gres.conf` (`gresConfig:`) | `File=/dev/nvidia0 ...` | **Which device file is that GPU** (the node's view) |

### gres.conf line, field by field

| Field | Meaning |
|---|---|
| `AutoDetect=off` | Don't query the driver (NVML). Trust this line. |
| `Name=gpu` | Resource name |
| `Type=nvidia_h100_80gb_hbm3` | Label that `--gres=gpu:<type>:N` must match |
| `File=/dev/nvidia0` | The device file. Slurm gives it to the job and checks it exists. |
| `Flags=nvidia_gpu_env` | Slurm sets `CUDA_VISIBLE_DEVICES` for the job |
| `Cores=` / `Links=` | CPU affinity + NVLink matrix. Only meaningful with 8 GPUs, so dropped in our 1-GPU override. |

### Where the hardware comes in

- The physical H100 → the NVIDIA driver on the K8s node creates `/dev/nvidia0` → it's exposed into the worker pod.
- Slurm checks the **device file exists** and the **count matches**. It does **not** check the model.
- The type string is a label. Our value matches what NVML autodetect would produce for "NVIDIA H100 80GB HBM3", so `--gres=gpu:nvidia_h100_80gb_hbm3:1` lines up with it.

### Why this matters: FINDINGS #3

The upstream module keys `gres_config` by **platform only**, so every H100 preset got the 8-GPU layout (`/dev/nvidia0-7`, cores 0–63). On our 1-GPU, 16-vCPU VM, `/dev/nvidia1-7` and cores 16–63 don't exist, so slurmd would fail or the node would go INVALID. Fix: a 1-GPU `gres_config` override in `installations/example/main.tf`.

### See the real files

Checked in the Soperator 4.1.12 source:
- `internal/render/common/configmap.go`: `RenderConfigMapSlurmConfigs` builds **one ConfigMap** with keys `slurm.conf`, `gres.conf`, `cgroup.conf`, `plugstack.conf`, `mpi.conf`, `slurm_rest.conf`, ...
- Same file, `RenderJailedConfigSlurmConfigs`: a `JailedConfig` object tells **sconfigcontroller** to copy those keys to `/etc/slurm/<file>` **inside the jail** (on the host side, `/mnt/jail/etc/slurm`), then run `scontrol reconfigure`.
- The jail is the shared filesystem every login and worker node sees, so all of them read the same files.

**Analogy:** Terraform writes the recipe → the operator prints it on a master sheet (the ConfigMap) → sconfigcontroller pins a copy on the shared kitchen noticeboard (`/etc/slurm` in the jail) → every cook (slurmd, slurmctld) reads the noticeboard.

```
Terraform values → Helm (flux) → SlurmCluster / NodeSet CRs
   → operator → ConfigMap <cluster>-slurm-configs   (Kubernetes)
   → sconfigcontroller → /etc/slurm/*.conf in the jail   (Slurm nodes read here)
```

**From the login node (simplest):**
```bash
cat /etc/slurm/gres.conf
grep -E '^NodeName|^NodeSet|^PartitionName|^GresTypes' /etc/slurm/slurm.conf
scontrol show config | grep -i -E 'gres|SlurmctldHost'    # what the running controller loaded
```

**From kubectl (the source copy):**
```bash
kubectl get configmap -A | grep slurm-configs
kubectl -n <ns> get configmap <cluster>-slurm-configs -o jsonpath='{.data.gres\.conf}'
kubectl -n <ns> get configmap <cluster>-slurm-configs -o jsonpath='{.data.slurm\.conf}' | grep NodeName
```

Expected `gres.conf` line for us (format from `AddNodesToGresConfig`): `NodeName=<nodeset>-[0-1] AutoDetect=off Name=gpu Type=nvidia_h100_80gb_hbm3 File=/dev/nvidia0 Flags=nvidia_gpu_env`

**Don't edit these files by hand.** The operator overwrites them. Change Terraform values instead.

### Verify on the cluster

```bash
scontrol show node | grep -E 'NodeName|Gres|State'     # what slurmctld believes
sinfo -o "%N %G"                                       # GRES per node
srun -N2 --gpus-per-node=1 nvidia-smi -L               # the real hardware
```
