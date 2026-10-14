# Study notes: LoRA, memory, Slurm, NCCL

Personal revision notes. Read them in order.

| # | File | Topic |
|---|---|---|
| 1 | [01-lora-basics.md](01-lora-basics.md) | What LoRA is, the chef-and-sticky-notes analogy, each parameter |
| 2 | [02-memory-math.md](02-memory-math.md) | Why a full 7B fine-tune needs ~112 GB and LoRA needs ~15 GB |
| 3 | [03-slurm-gres.md](03-slurm-gres.md) | `--gres=gpu:nvidia_h100_80gb_hbm3:1`, and sbatch vs srun |
| 4 | [04-training-pipeline-test.md](04-training-pipeline-test.md) | The 1.5B / 30-step test command and torchrun rendezvous |
| 5 | [05-nccl-test.md](05-nccl-test.md) | The 2-node NCCL all-reduce test, flag by flag |
| 6 | [06-why-lora-answer.md](06-why-lora-answer.md) | The interview answer to "Why did you choose LoRA?" |
| 7 | [07-soperator-filesystem.md](07-soperator-filesystem.md) | The shared "jail" filesystem: chroot, SPANK, submounts, our storage layout |
| 8 | [08-attention-vs-mlp.md](08-attention-vs-mlp.md) | Attention vs MLP layers, and which ones LoRA should target |
| 9 | [09-ddp-vs-fsdp.md](09-ddp-vs-fsdp.md) | DDP vs FSDP: where the network traffic happens, layer by layer, with the chef analogy |

## The one-liners to remember

- **LoRA:** "Frozen cookbook, sticky notes. r is the note size, alpha is the volume, dropout hides a few, target_modules picks the pages."
- **Memory:** "Full fine-tune = 16 bytes per parameter → 7B × 16 = ~112 GB. LoRA = 14 GB frozen + ~0.6 GB trainable."
- **Network:** "Full fine-tune syncs ~14 GB per step (~13 s at 1.1 GB/s). LoRA syncs ~100 MB (~0.1 s)."
- **Slurm:** "sbatch reserves, srun runs inside the reservation. `--gres` counts are per node."
- **Layers:** "Attention = listen to the room. MLP = think at your desk. LoRA on all-linear = coach both."
- **DDP vs FSDP:** "DDP chefs own the whole book and swap notes once. FSDP chefs keep half the pages and photocopy the rest for every chapter, twice."
- **Jail:** "One shared office, many doors, personal desk drawers. The building's machinery is off limits."
- **NCCL test:** "An empty truck on the route before the moving day. Use `_mpi` + `--mpi=pmix`, or you get two separate 1-GPU tests."
