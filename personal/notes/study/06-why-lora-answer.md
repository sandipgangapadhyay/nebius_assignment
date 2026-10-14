# 6. "Why did you choose LoRA?" (interview answer)

## The 20-second infra version (no detailed numbers)

> "Full fine-tuning costs about 16 bytes per parameter, because you hold the weights plus gradients and optimizer state for every one of them. For a 7B model that's over 100 GB before activations, more than one 80 GB H100. With LoRA, the base model is frozen: about 2 bytes per parameter, roughly 14 GB. We only train small adapter matrices, under 1% of the parameters, so gradients and optimizer state are a few hundred MB. It fits on one GPU, and only the small adapter gradients cross the network."

Rules to keep straight: **16 bytes** per param (not "16 times"), frozen model = **2 bytes** per param, adapter overhead = **a few hundred MB** (not "a few MB"), adapters go beside **every** linear layer (not "selected layers").

## The ~45-second answer

> "I chose LoRA mainly for memory and network reasons.
>
> A full fine-tune of a 7-billion-parameter model with Adam needs about 16 bytes per parameter: the bf16 weights, the gradients, an fp32 master copy and two optimizer states. That's around 112 GB before activations, so it doesn't fit on one 80 GB H100.
>
> LoRA freezes all the original weights. It adds small low-rank adapter matrices beside every linear layer and trains only those, about 40 million parameters, under 1% of the model. Gradients and optimizer state exist only for those, so memory drops to about 15 GB plus activations, which fits on one H100 per node.
>
> It also matters for the network. Our nodes talk over TCP at about 1.1 GB/s. A full fine-tune would sync about 14 GB of gradients every step. LoRA syncs about 100 MB, a tenth of a second.
>
> The idea behind it is that adapting a model to a task like tool calling needs only a small, low-rank change, not a rewrite of every weight. The trade-off is that a big domain shift might still justify a full fine-tune."

**One line:** *"Freeze the 7B, train a 40M low-rank add-on. Memory ~112 GB → ~15 GB, network sync 14 GB → ~100 MB per step."*

## Mistakes to avoid

| Don't say | Say instead |
|---|---|
| "seven million parameters" | "**7 billion**" |
| "needs two times the memory" | "~16 bytes per parameter, ~**8×** the weights, ~112 GB" |
| "it selects a few layers and updates their parameters" | "it freezes **all** weights and **adds** small low-rank matrices beside **every** linear layer" (picking layers = partial fine-tuning, a different technique) |
| "gradients/optimizer aren't duplicated" | "gradients and optimizer state exist **only for the ~40M adapter params**" |
| "15 GB at full precision" | "~15 GB **in bf16, before activations**" |
| "15 GB total" | "15 GB + activations (batch × seq length) + CUDA/NCCL overhead" |

## Extra points if there's time

- **Network:** the strongest point for this cluster (1.1 GB/s TCP, no InfiniBand).
- **Small output:** adapter ~100 MB vs 15 GB, and vLLM loads it on top of the base model.
- **Base model preserved:** less catastrophic forgetting.
- **Fits the task:** tool calling is teaching a format/behaviour, which is where LoRA works well.
- **Trade-off:** a big domain shift (e.g. new language) can justify a full fine-tune.

## Likely follow-ups

- **"Is LoRA faster?"** Somewhat (~1.2–1.5× per step). The forward pass costs the same. The real win is memory, which allows bigger batches, plus smaller DDP traffic.
- **"Why r=16, alpha=32?"** Common default. r = adapter capacity; alpha/r = 2 is the scaling. Bigger r = more capacity, more overfitting risk.
- **"Why all-linear?"** Attention + MLP both get adapters → better fit for small extra memory.
- **"How would you go further on memory?"** QLoRA: 4-bit frozen base (14 GB → ~4 GB). Or 8-bit Adam.
