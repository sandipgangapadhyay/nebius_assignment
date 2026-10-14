# 1. LoRA basics

## What it is

A pretrained model (e.g. 7B LLM) is billions of numbers in big weight matrices.

- **Full fine-tune:** update every one of them.
- **LoRA (Low-Rank Adaptation):** freeze all of them. Add a pair of small matrices (A and B) beside each layer and train only those. Their output is added to the frozen layer's output.

```
output = W·x  +  (alpha/r) · B·A·x
         ↑ frozen    ↑ the only part that trains
```

**Why it works:** the *change* a model needs to learn a new task is usually simple (low-rank). Two thin matrices can represent it.

**Numbers for one 4096×4096 layer:**
- Full: 4096 × 4096 = **16.7M** trainable
- LoRA r=16: (4096×16) + (16×4096) = **131K**, about **0.8%**

**B starts at zero**, so on day one the adapter does nothing and the model behaves exactly like the base. It learns gradually.

## Analogy: master chef + sticky notes

- **Base model = master chef** with a huge cookbook (the weights). Already cooks almost everything.
- You want *your restaurant's style*.
  - **Full fine-tune** = rewrite the whole cookbook. Slow, expensive, can ruin recipes that worked.
  - **LoRA** = leave the cookbook alone, **put sticky notes on the pages** ("less salt", "add chili"). The chef reads recipe + note.

## Parameters (from `poc/training/train.py`)

```python
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
```

| Parameter | Value | Analogy | What it really does |
|---|---|---|---|
| `r` (rank) | 16 | **Size of each sticky note** | Width of A/B. Bigger = more detail, more memory, easier to overfit. Typical 8–64. |
| `lora_alpha` | 32 | **How loudly the chef listens to the notes** | Output scaled by `alpha/r` = 32/16 = 2. Volume knob. Convention: 2 × r. |
| `lora_dropout` | 0.05 | **Hide 5% of notes at random during practice** | Regularisation, so it doesn't over-rely on any one note. Training only. |
| `target_modules` | `"all-linear"` | **Which pages get notes** | Attention layers (what the model looks at) + MLP layers (stored knowledge). More pages = better fit, a bit more memory. |
| `task_type` | `CAUSAL_LM` | **What kind of cooking** | Next-token prediction. Tells PEFT how to wrap the model. |

**Memory hook:** *"Frozen cookbook, sticky notes. r is the note size, alpha is the volume, dropout hides a few, target_modules picks the pages."*

## What you gain

1. **Memory (biggest win).** ~112 GB → ~15 GB for 7B (see [02-memory-math.md](02-memory-math.md)). Fits on 1 GPU.
2. **Speed: only moderate.** Forward pass costs the same. Backward skips weight gradients for the frozen part, optimizer step is tiny. ~1.2–1.5× per step, not 10×. Bigger win: spare memory → larger batches.
3. **Small DDP traffic.** Only adapter gradients are all-reduced: ~MBs instead of ~14 GB per step.
4. **Small checkpoints.** Adapter ~50–200 MB vs ~14 GB. One base model, swap adapters per task.
5. **Less forgetting.** Base weights are never overwritten.

**Trade-off:** for a big domain shift (e.g. a new language), a full fine-tune can still give better quality.

## Other ways to picture it

- **Photo filter:** the photo stays, the filter changes the look. Remove it to get the original back.
- **Game mod / DLC:** small download that changes behaviour, no reinstall.
- **GPS overlay:** the map is fixed, LoRA draws your route on top.
- **Math view:** the *update* to the weights is low-rank. You learn a compressed delta, not new weights.

## Common mistake to avoid

LoRA does **not** pick a few layers and unfreeze them. That's a different technique (partial fine-tuning / layer freezing). LoRA changes **no** existing weights. It **adds** new small matrices, and with `all-linear` it adds them beside **every** linear layer.
