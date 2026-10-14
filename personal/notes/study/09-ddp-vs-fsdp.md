# 9. DDP vs FSDP: where the network traffic happens

Full fine-tune of Qwen2.5-7B (28 decoder layers, ~15 GB of bf16 weights) on 2 GPUs over ~1.1 GB/s Ethernet, unless stated otherwise.

## One-line summary

- **DDP:** every GPU holds the whole model and **talks once per step** (gradients), overlapped with compute.
- **FSDP:** every GPU holds **half** of everything and has to **fetch the other half for every layer**, twice. It saves memory and costs network.

## DDP: each GPU holds the full model

```
Forward pass   ─ layer 1 → 28        no network (all weights are local)
Backward pass  ─ layer 28 → 1        computes gradients layer by layer
                  └── ALL-REDUCE gradients in buckets as they're ready (~15 GB),
                      in the background while backward continues
Optimizer      ─ Adam on the full model, no network
```

- Every GPU stores the full **parameters, gradients and optimizer state** (fp32 master weights + Adam moments) ≈ 16 bytes/param ≈ **~120 GB** for a 7B full fine-tune. **It doesn't fit in 80 GB.**
- After the all-reduce, both GPUs apply the same averaged gradients, so the replicas stay identical.
- **With gradient accumulation 2:** forward + backward on 2 micro-batches, all-reduce **only after the 2nd**, then one optimizer step. One sync per update: **592 syncs** in our run, not 1,184.

## FSDP (`FULL_SHARD`): each GPU holds half of every layer

```
Forward pass   ─ for each layer 1 … 28:
                   ALL-GATHER parameters (fetch the missing half)   ← GPU waits
                   compute, keep activations locally
                   free the gathered half
Backward pass  ─ for each layer 28 … 1:
                   ALL-GATHER parameters AGAIN (they were freed)    ← GPU waits
                   compute gradients
                   REDUCE-SCATTER gradients (average; each GPU keeps its shard)
Optimizer      ─ Adam on your own shard only, no network
```

- Each GPU stores ~half the parameters, gradients and optimizer state ≈ **~61 GB** + activations. It fits, but only just.
- **Activations never cross the network.** Only parameters (all-gather) and gradients (reduce-scatter) do.
- Prefetching the next layer's weights hides some waiting, but only on a fast network.

## Traffic per GPU per step

| Collective | When | What | ~Bytes |
|---|---|---|---|
| All-gather | Forward, per layer | Parameters | ~7.5 GB |
| All-gather | Backward, per layer | Parameters, again | ~7.5 GB |
| Reduce-scatter | Backward, per layer | Gradients | ~7.5 GB |
| **FSDP total** | | | **~23 GB → ~21 s** |
| DDP all-reduce | Backward, overlapped | Gradients | ~15 GB → ~14 s |
| **DDP + LoRA (ours)** | Backward, overlapped | Adapter gradients only | **~0.15 GB → ~0.1 s** |

That's **three transfers, but ~1.5× the bytes of DDP, not 3×**, because each FSDP transfer moves only the half a GPU is missing.

## Comparison

| | DDP + LoRA (ours) | DDP, full fine-tune | FSDP, full fine-tune |
|---|---|---|---|
| Each GPU holds | Full frozen model + small adapter | Everything | Half of everything |
| Memory per GPU | ~40 GB measured ✅ | ~120 GB ❌ | ~61 GB + activations ⚠️ |
| Network per step | ~0.15 GB | ~15 GB | ~23 GB |
| Blocks compute? | No (overlapped) | Mostly overlapped | **Yes, every layer waits** |
| GPU utilisation | **96–100 %** | n/a (doesn't fit) | likely < 30 % |

## Analogy: two chefs, one 28-chapter recipe book

| Analogy | Real term |
|---|---|
| Recipe book chapter | Decoder layer |
| Half the pages each chef keeps | Parameter shard |
| Photocopying the missing pages | All-gather of parameters |
| Shredding the copy | Freeing the gathered weights |
| Correction notes | Gradients |
| Combining notes, each keeping their half | Reduce-scatter of gradients |
| Editing your own pages | Optimizer step on your shard (Adam) |
| DDP chefs swapping notes once | All-reduce of gradients |
| Sticky note | LoRA adapter gradients |

- **FSDP:**
  - **Cooking (forward):** for every chapter, the chefs photocopy each other's missing pages, cook, then shred the copy.
  - **Tasting (backward):** they photocopy the same pages again, write correction notes, and combine them; each files only the notes for their own pages.
  - **Update:** each chef edits only their own pages.
  - With a slow courier (Ethernet), they mostly stand around waiting for photocopies.
- **DDP:** each chef owns the whole book, so there's no photocopying. They swap correction notes once, while still cooking. **With LoRA, the notes are a sticky note.**

## Why we chose DDP

1. FSDP exists to **save memory**. LoRA already makes the model fit on one GPU (~40 GB of 80).
2. FSDP **costs bandwidth**, the scarcest resource here (no InfiniBand, ~1.1 GB/s).
3. With 1 GPU per node there's no fast NVLink to shard over, so every shard boundary is a network hop.
4. It's simpler: no wrap policy and no sharded checkpoints.

**When to switch to FSDP:** full fine-tuning or models above ~13B, on 8-GPU InfiniBand nodes with `HYBRID_SHARD` (shard over NVLink inside the node, replicate across nodes).

## If we did use FSDP

- `train.py`: add `fsdp="full_shard auto_wrap"`, `fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen2DecoderLayer"]}`, and `use_orig_params=True` for LoRA.
- `train.sbatch`: no change (same `srun torchrun` on 2 nodes).
- Checkpoints become per-GPU shards, which need merging or a full-state-dict setting.
