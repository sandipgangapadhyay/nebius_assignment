# 4. Training pipeline test and torchrun rendezvous

## The command

```bash
cd ~/training && sbatch --export=ALL,MODEL=/data/models/Qwen2.5-1.5B-Instruct,RUN=test-1.5b,MAX_STEPS=30 train.sbatch
```

**Purpose:** check the whole pipeline cheaply (~5 min) before the 1–2 h 7B run: Slurm, both nodes, GPUs, inter-node network, data loading, LoRA, checkpoint saving.

| Part | Meaning |
|---|---|
| `cd ~/training` | Folder with `train.sbatch` and `train.py` |
| `sbatch ... train.sbatch` | Submit as a batch job, get a job ID back immediately |
| `--export=ALL,...` | Pass env vars into the job. `ALL` keeps the current shell's vars, the rest add/override |
| `MODEL=.../Qwen2.5-1.5B-Instruct` | Small model instead of the 7B default → fast |
| `RUN=test-1.5b` | Run name → output in `/data/checkpoints/test-1.5b/`. Required. |
| `MAX_STEPS=30` | Stop after 30 updates (default `-1` = full 2 epochs, ~590 updates) |

## What happens

```
sbatch  ──►  Slurm reserves 2 nodes × 1 H100
   │
   ├── activate venv, read MODEL / RUN / MAX_STEPS
   ├── HEAD_NODE = first node, MASTER_ADDR = its IP
   ├── NCCL over TCP on eth0 (no InfiniBand)
   │
   └── srun torchrun ...   → one torchrun per node
          ├── node 1: train.py on GPU 0  ─┐
          └── node 2: train.py on GPU 0  ─┴─ meet at MASTER_ADDR:29500 → DDP
```

## Rendezvous

```bash
--rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" --rdzv_id="${SLURM_JOB_ID}"
```

"Rendezvous" = the **meeting point** where the nodes find each other before training.

| Flag | Meaning | Analogy |
|---|---|---|
| `--rdzv_backend=c10d` | PyTorch's built-in meeting service, hosted on the head node (no etcd) | Meeting format: "at a café" |
| `--rdzv_endpoint=IP:29500` | Where to meet | The café's address |
| `--rdzv_id=$SLURM_JOB_ID` | Unique meeting name, so two jobs never mix | Booking name: "table for job 42" |

After both check in, torchrun assigns ranks (0, 1). DDP starts, and after each step the GPUs average their LoRA gradients.

## Success looks like

- `tail -f train-lora-<ID>.out`: loss every 10 steps, trending down
- Job ends cleanly after step 30
- `/data/checkpoints/test-1.5b/adapter` exists

## Then the real run

```bash
sbatch --export=ALL,RUN=qwen7b-lora train.sbatch
```
