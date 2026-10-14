# 8. Attention vs MLP layers, and which ones LoRA should target

## The big picture

A transformer (like Qwen2.5-7B) is a stack of identical **blocks**. Qwen2.5-7B has **28**. Each block has two parts, always in this order:

```
text → [ Attention → MLP ] → [ Attention → MLP ] → ... (28 times) → next word
```

## Analogy: a team meeting, repeated 28 times

Each word in the prompt is a **person in a meeting**.

**1. Attention = listening to the room.**
Each person asks: *"Who else in this room matters to me right now?"* and pulls in what they say.
- The word "it" looks back and finds that "it" means "the weather API".
- The user's request "book a flight to Paris" connects to the `book_flight` tool definition 2,000 tokens earlier.

It's about **relationships between words**: who refers to whom, what goes with what.

**2. MLP = going back to your desk and thinking alone.**
Each person now works **on their own**, using **what they already know** (facts, rules, patterns stored in the weights), and updates their understanding. No looking at others.
- "Paris" → it's a city, a destination, likely needs an airport code.
- "I'm inside a tool call" → the arguments must be valid JSON, strings quoted.

It's about **knowledge and processing**: facts, formats, rules.

**Repeat 28 rounds:** listen, think, listen, think... Each round the understanding gets richer. At the end, the last person predicts the next word.

**Memory hook:** *"Attention = listen to the room. MLP = think at your desk."*

## The parts inside each (the names you see in LoRA configs)

| Part | Layers | Plain English |
|---|---|---|
| **Attention** | `q_proj` (query) | "What am I looking for?" |
| | `k_proj` (key) | "What do I offer to others?" (like a name tag) |
| | `v_proj` (value) | "What I actually tell someone who listens to me" |
| | `o_proj` (output) | Combine what I heard into one update |
| **MLP** | `gate_proj`, `up_proj` | Expand into a much bigger "thinking space" (3584 → 18944) and decide what to keep |
| | `down_proj` | Compress back to the normal size (18944 → 3584) |

## Differences at a glance

| | Attention | MLP |
|---|---|---|
| Question it answers | "Which other words matter?" | "What do I know about this?" |
| Looks at other words? | ✅ Yes, mixes information **across** words | ❌ No, each word **separately** |
| Holds | Relationships, context, references | Knowledge, facts, rules, formats |
| Share of weights (Qwen2.5-7B, per block) | ~13% | **~87%** |
| Cost grows with prompt length? | Yes (every word looks at every other word) | No (same cost per word) |

## Which layers should LoRA target?

| Option | Layers | Adapter size (Qwen2.5-7B, r=16) | When to use |
|---|---|---|---|
| **Attention only (q, v)** | `q_proj`, `v_proj` | ~5M params, ~20 MB | Original LoRA paper. Cheapest. OK for light style/tone changes. |
| **All attention** | q, k, v, o | ~10M | Change how the model connects context |
| **All linear (attention + MLP)** ✅ ours | all 7 | ~40M params, ~155 MB | Current best practice. The QLoRA paper found you need all layers to match full fine-tune quality. Best when learning new rules or formats. |
| MLP only | gate, up, down | ~30M | Rare; mostly for injecting knowledge |

**Why `all-linear` for our tool-calling task:**
- **Attention** must learn to link the user's request to the right tool schema, often far back in a long prompt, and decide how many calls to make (parallel calls).
- **MLP** must learn the rules: exact JSON format, don't add unrequested optional arguments, when *not* to call a tool.
- Both matter, and the extra cost is tiny: ~155 MB adapter vs ~20 MB, still under 1% of the model, still ~0.6 GB of training memory.

**Evidence from our run:** the biggest improvements were rule-like behaviours ("extra unrequested arguments" 120 → 31, "correctly answered without a tool" 60 → 78%). That's the kind of thing the MLP adapters help with.

## Interview one-liner

> "Each transformer block has attention, where words look at each other to understand context, and an MLP, where each word is processed using the model's stored knowledge. Most weights are in the MLP. The original LoRA paper only adapted attention, but current practice, and the QLoRA results, show targeting all linear layers works better. For tool calling we need both: attention to match the request to the right tool, MLP to learn the formatting rules. It only costs about 40M parameters instead of 5M, still under 1% of the model."
