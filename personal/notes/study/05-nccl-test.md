# 5. NCCL test: 2 nodes over TCP

## The command

```bash
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
srun -N2 --ntasks-per-node=1 --gpus-per-node=1 --mpi=pmix all_reduce_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -n 20 -w 5 -c 1 2>&1 | tee ~/nccl-2node.log
```

## What it tests

Can the two GPUs on two nodes **talk** through NCCL, and **how fast**? DDP does an **all-reduce** after every step to average gradients. This runs all-reduce alone, no model, so a network problem shows up in 2 minutes instead of mid-training.

**Analogy:** drive an empty truck along the route and time it with bigger and bigger loads before moving day.

## Env vars

| Variable | Meaning |
|---|---|
| `NCCL_IB_DISABLE=1` | No InfiniBand on this hardware |
| `NCCL_SOCKET_IFNAME=eth0` | Use eth0. Otherwise NCCL may pick `docker0` (down, same IP on every node) |
| `NCCL_DEBUG=INFO` | Print what NCCL does |
| `NCCL_DEBUG_SUBSYS=INIT,NET` | Only start-up and network logs → confirm `NET/Socket` on eth0 |

`srun` forwards the shell env to both nodes by default.

## srun part

| Part | Meaning |
|---|---|
| `-N2` | 2 nodes, srun reserves + runs (no sbatch) |
| `--ntasks-per-node=1` | 1 process per node → 2 ranks |
| `--gpus-per-node=1` | Each process gets the node's H100 |
| `--mpi=pmix` | Slurm starts processes as MPI ranks via PMIx ("rank 0 of 2", "rank 1 of 2") |
| `all_reduce_perf_mpi` | nccl-tests all-reduce benchmark, MPI build |

**Why `_mpi` matters:** plain `all_reduce_perf` can't find the other node → **two separate 1-GPU tests**, useless. `_mpi` + `--mpi=pmix` → **one 2-rank test**.

## Benchmark flags

| Flag | Meaning |
|---|---|
| `-b 8` | Start at 8 bytes |
| `-e 1G` | End at 1 GB |
| `-f 2` | Double each round |
| `-g 1` | 1 GPU per process |
| `-n 20` | 20 timed iterations per size |
| `-w 5` | 5 warm-up iterations |
| `-c 1` | Check correctness (`#wrong` must be 0) |
| `2>&1 \| tee ~/nccl-2node.log` | Screen + file |

Quicker run: `-e 256M`.

## Reading the result

- **algbw** = size ÷ time
- **busbw** = algorithm-adjusted, comparable across GPU counts. With 2 ranks, busbw = algbw.
- **#wrong** = must be 0

**Expected here: busbw ≈ 1.1 GB/s** (~9 Gbit/s), normal for TCP Ethernet. InfiniBand would be tens of times faster.

```bash
grep -v "NCCL INFO" ~/nccl-2node.log               # results table
grep -E "Using network|Avg bus" ~/nccl-2node.log   # Socket/eth0 + average busbw
```

## Why 1.1 GB/s is fine for LoRA

| | Sync per step | Time |
|---|---|---|
| LoRA (~40M params) | ~80–160 MB | ~0.1 s |
| Full fine-tune (7B) | ~14 GB | ~13 s |

LoRA is what makes 2-node training over plain Ethernet practical.
