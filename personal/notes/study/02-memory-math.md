# 2. Memory math: why ~112 GB vs ~15 GB

## Rule of thumb: 16 bytes per trainable parameter

Mixed-precision training with Adam (from the ZeRO paper, Microsoft DeepSpeed):

| What GPU memory holds | Precision | Bytes | Why |
|---|---|---|---|
| Model weights | bf16 | 2 | Forward and backward pass |
| Gradients | bf16 | 2 | One per weight |
| Master copy of weights | fp32 | 4 | Tiny updates need full precision to avoid rounding errors |
| Adam momentum (m) | fp32 | 4 | Running average of gradients ("which direction") |
| Adam variance (v) | fp32 | 4 | Running average of squared gradients ("how bumpy") |
| **Total** | | **16** | |

## Full fine-tune, 7B

```
7,000,000,000 × 16 bytes ≈ 112 GB     (before activations)
```

That's about **8× the weights alone** (14 GB), not 2×. Doesn't fit on one 80 GB H100 → needs several GPUs + FSDP/ZeRO sharding.

**Activations** come on top: values saved in the forward pass for the backward pass. They grow with batch size × sequence length and can add tens of GB.

## LoRA, 7B (r=16, all-linear, Llama-style, 32 layers)

**Frozen weights:** weights only, no gradients, no optimizer state.
```
7B × 2 bytes (bf16) = 14 GB
```

**Trainable adapter parameters (our actual model, Qwen2.5-7B, r=16, all-linear):**

Rule for one LoRA pair on a layer of shape `in → out`: **r × (in + out)** (A is r×in, B is out×r).

Qwen2.5-7B: hidden 3584, MLP 18944, 28 layers, k/v project to 512 (grouped-query attention).

| Layer | Shape | r × (in + out) |
|---|---|---|
| q_proj | 3584 → 3584 | 16 × 7168 = 114,688 |
| k_proj | 3584 → 512 | 16 × 4096 = 65,536 |
| v_proj | 3584 → 512 | 16 × 4096 = 65,536 |
| o_proj | 3584 → 3584 | 16 × 7168 = 114,688 |
| gate_proj | 3584 → 18944 | 16 × 22528 = 360,448 |
| up_proj | 3584 → 18944 | 16 × 22528 = 360,448 |
| down_proj | 18944 → 3584 | 16 × 22528 = 360,448 |
| **Per layer** | | **1,441,792** |

× 28 layers = **~40.4M** trainable parameters (~0.5% of 7.6B).

```
40.4M × 16 bytes ≈ 646 MB ≈ 0.6 GB
```

The 16 bytes for the adapter: PEFT keeps LoRA weights in **fp32**, so it's 4 (weights) + 4 (gradients) + 4 + 4 (Adam m, v), with no separate master copy needed. Still 16.

**Proof from our run:** the saved adapter is **155 MB** = 40.4M × 4 bytes (fp32) ≈ 161 MB ≈ 154 MiB. ✅

## Side by side

| | Full fine-tune | LoRA |
|---|---|---|
| Weights | 14 GB | 14 GB (frozen) |
| Gradients + fp32 master + Adam | ~98 GB | ~0.6 GB |
| **Total before activations** | **~112 GB** | **~15 GB** |

**Key insight:** the savings are not in the weights (both hold 14 GB). LoRA removes the **training bookkeeping**, which costs 14 extra bytes for every parameter you train.

Say "~15 GB **before activations**, in **bf16**", not "15 GB total" and not "full precision".

## "We have 2 × 80 GB = 160 GB. Why not full fine-tune?"

**Two GPUs are not one 160 GB GPU.** Memory isn't pooled automatically. Each GPU only sees its own 80 GB.

**Analogy:** two 80-litre suitcases don't hold a 112-litre wardrobe unless you split the clothes between them, and then you keep passing items back and forth while getting dressed.

| Approach | Per-GPU memory | Works? |
|---|---|---|
| **DDP** (what we use) | Full copy on each GPU → ~112 GB each | ❌ Doesn't fit |
| **FSDP / ZeRO-3** (shard everything across 2 GPUs) | ~56 GB + activations + gather buffers + CUDA overhead | ⚠️ Very tight. Needs gradient checkpointing and small batches. |
| **FSDP + 8-bit Adam / CPU offload** | Lower (~35–45 GB) | ✅ Fits, but slower |

**The real blocker is the network, not memory.** Sharding means the GPUs constantly swap weight pieces. Rough estimate per step with FSDP: all-gather weights for forward + all-gather for backward + reduce-scatter gradients ≈ 3 × 14 GB of traffic, about half crossing the link with 2 ranks → **~20 GB per step**.

At our measured **~1.1 GB/s** (TCP, no InfiniBand): **~20 s+ per step just on communication**, × ~590 steps ≈ **3+ hours of waiting**, vs ~1–2 h for the whole LoRA run. (On InfiniBand, ~40+ GB/s, this would be fine.)

**Answer to give:** "Memory-wise it's borderline possible with FSDP, but two GPUs don't pool memory automatically, so we'd need full sharding. That makes the GPUs exchange ~20 GB per step over a 1.1 GB/s link, so communication dominates. For a tool-calling format task, LoRA gets most of the quality at a fraction of the cost. With InfiniBand or 8-GPU nodes I'd consider a full fine-tune."

## Variants

- **8-bit Adam / bf16 optimizers:** lower than 16 bytes per param.
- **QLoRA:** frozen base in 4-bit → 14 GB drops to ~4 GB.

## Network side (DDP)

DDP all-reduces only the **trainable** gradients each step. Measured inter-node speed: ~1.1 GB/s (TCP, eth0).

| | Gradients per step | Time at 1.1 GB/s |
|---|---|---|
| Full fine-tune | ~14 GB | ~13 s |
| LoRA | ~80–160 MB | ~0.1 s |
