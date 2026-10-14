# 2. Fine-tuning and inference: Qwen2.5-7B for function calling

## Goal

- **Exercise 1:** teach an LLM to **call tools reliably**: pick the right function, fill the arguments correctly, make parallel calls, and know when *not* to call. Run it as a **distributed Slurm job across 2 nodes × 1 H100**, with **> 80 % GPU utilisation**.
- **Exercise 2:** **serve** the fine-tuned model on the same cluster next to the original model, and compare the two.

## Approach at a glance

| | |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0). Qwen2.5-1.5B for a quick pipeline test |
| Method | LoRA fine-tuning (base model frozen, small adapter trained) |
| Distribution | PyTorch DDP across 2 nodes × 1 H100, NCCL over Ethernet (`eth0`) |
| Stack | Hugging Face TRL `SFTTrainer` + PEFT, launched with `srun torchrun` |
| Dataset | `Team-ACE/ToolACE` (Apache-2.0), cleaned and validated: 10,514 conversations |
| Split | 90 % train / 5 % validation / 5 % test (725 scored test turns, never seen in training) |
| Serving | vLLM as a Slurm job on 1 GPU; one server answers as both the base model and the fine-tuned model |
| Storage | Model, data, Python environments and checkpoints on the shared 1 TiB jail filesystem |

Commands to reproduce everything: [03-reproduce.md](03-reproduce.md).

## Training results (7B)

| | 2 epochs (job 44) | 3 epochs (job 379) |
|---|---|---|
| Run time | 43 min 44 s, 592 steps | 1 h 05 min, 888 steps |
| Throughput | 7.5 conversations/s | 7.5 conversations/s |
| GPU utilisation | 96–100 % on both nodes | ~100 % (worker-0, W&B) |
| Train loss | 1.54 → 0.43 | 1.59 → 0.38 |
| Validation loss | 0.536 → 0.433 | 0.535 → 0.409 |

- The target was > 80 % GPU utilisation. GPU memory used: ~40–45 GB of 80 GB.
- The adapter is only **155 MB**. Checkpoints are saved every 100 steps, and the job auto-resumes from them.

**Charts from the 3-epoch run (Weights & Biases)**

**Training loss**

![Training loss](../imgs/train-loss.png)

- Drops fast in the first ~30 steps as the model picks up the tool-call format, then declines slowly to ~0.38. The dark line is smoothed.

**Validation loss**

![Validation loss](../imgs/eval-loss.png)

- Measured on data the model never trains on. It falls at every check and levels off near 0.41, so the model is learning, not memorising.

**Learning rate**

![Learning rate](../imgs/learning%20rate.png)

- Short warm-up, then cosine decay to 0, which is why both loss curves flatten at the end.

**GPU memory and utilisation (worker-0)**

![GPU memory and utilisation](../imgs/gpu-util.png)

- GPU utilisation stays close to **100 % for the whole 64-minute run**. The short dips are the periodic validation and checkpoint pauses.

**Nebius console (worker-1, last ~12 minutes of the run)**

| Used frame buffer | Memory utilization |
|---|---|
| ![Used frame buffer](../imgs/nebius-gpu.png) | ![Memory utilization](../imgs/nebius-gpu-util.png) |

- **Used frame buffer:** ~45 GB of GPU memory in use until training ends.
- **Memory utilization** is GPU memory *bandwidth*, busy 60–80 % of the time (not memory used).

## Inference: serving both models

- **One vLLM server on one GPU**, run as a Slurm job, with the LoRA adapter loaded on top of the base model. Each request picks a model by name:
  - `Qwen2.5-7B-Instruct`: the original, untrained model;
  - `toolace`: the fine-tuned model.
- It's an **OpenAI-compatible API**, and Qwen's tool calls are returned as standard `tool_calls`.
- **Only one copy of the 15 GB base weights is needed.** Model and adapter are read straight from the shared filesystem.
- **To run it:** submit `serve.sbatch` with `sbatch`, then run `chat_compare.py` for a side-by-side answer. See [03-reproduce.md](03-reproduce.md#c-inference-base-vs-fine-tuned).

## Comparison: base vs fine-tuned

- Both models answer the same **725 held-out test turns**, using greedy decoding and the same prompts.
- A scorer compares their tool calls with the reference answers.

| Metric | Base | Fine-tuned, 2 epochs | Fine-tuned, 3 epochs |
|---|---|---|---|
| **Exact match** (name + all arguments) | 56.4 % | **68.2 %** (+11.8) | **68.2 %** (+11.8) |
| … single call / parallel calls | 67.3 / 40.2 % | 79.9 / 50.8 % | 79.6 / 51.3 % |
| Right tool chosen | 87.6 % | 91.1 % | 90.9 % |
| **Correctly answered without a tool** | 60.3 % | **78.0 %** (+17.7) | **80.2 %** (+19.9) |
| Malformed tool call | 0.4 % | 0.4 % | 0.2 % |
| Validation loss | – | 0.433 | 0.409 |
| Training time | – | 44 min | 65 min |

- **3 epochs vs 2:** slightly better (lower validation loss, +2.2 pts on answering without a tool, fewer malformed calls) for ~50 % more training time. Exact match is unchanged.

- **What improved:**
  - Calls with unrequested extra arguments fell from **120 to 31**.
  - The model stopped calling tools after already giving the answer.

**Live side-by-side example** (`toolace-00488#7`): the user asks for the latest CO₂ levels **and** data for 5 random users.

| | Answer | Result |
|---|---|---|
| Expected | `Carbon Dioxide Endpoint()` + `Random User Copy(num_users=5)` | |
| Base | `Carbon Dioxide Endpoint()` only | ❌ misses the 2nd call |
| Fine-tuned | `Carbon Dioxide Endpoint()` + `Random User Copy(num_users=5)` | ✅ matches |

## Challenges

- **The dataset had incorrect labels.** A "cleaned" copy on Hugging Face had missing arguments and wrong tool calls, so the data was rebuilt from the original ToolACE and checked to match the model's expected format.
- **The two nodes tried to connect over the wrong network interface** (`docker0`, which is down, instead of `eth0`). Fixed by resolving each node's `eth0` address explicitly.

## Next steps

- Tune further hyperparameters one change at a time (e.g. learning rate, LoRA rank), comparing by validation loss.
- Scale the same scripts to 8-GPU InfiniBand nodes for the full reservation.
