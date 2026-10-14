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

## Training results (7B, job 44)

- **43 min 44 s**, 592 steps, 7.5 conversations/s.
- **GPU utilisation 96–100 % on both nodes** (target > 80 %), ~40 GB of 80 GB GPU memory used.
- Train loss 1.54 → 0.43. **Validation loss fell at every check (0.536 → 0.433)**, so no overfitting.
- The adapter is only **155 MB**. Checkpoints are saved every 100 steps, and the job auto-resumes from them.

**Training loss**

![Training loss](../imgs/train-loss.png)

- Drops fast in the first ~30 steps (1.5 → 0.6) as the model picks up the tool-call format, then declines slowly to ~0.42.

**Validation loss**

![Validation loss](../imgs/eval-loss.png)

- Measured on data the model never trains on. It falls at every check and levels off near the end, so the model is learning, not memorising.

**Learning rate**

![Learning rate](../imgs/learning%20rate.png)

- Short warm-up, then cosine decay to 0, which is why both loss curves flatten at the end.

**GPU utilisation (worker-0)**

![GPU utilisation](../imgs/gpu-util.png)

- Close to **100 % for the whole run**. The short dips are the periodic validation and checkpoint pauses.

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

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| **Exact match** (name + all arguments) | 56.4 % | **68.2 %** | **+11.8 pts** |
| … single call / parallel calls | 67.3 / 40.2 % | 79.9 / 50.8 % | +12.6 / +10.6 |
| Right tool chosen | 87.6 % | 91.1 % | +3.5 |
| **Correctly answered without a tool** | 60.3 % | **78.0 %** | **+17.7** |
| Malformed tool call | 0.4 % | 0.4 % | 0 |

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

- **The data needed clean-up to get correctly formatted training examples:** the "cleaned" Hugging Face copy had wrong labels, and tool-call arguments were double-encoded (quoted JSON strings). Rebuilt from the original ToolACE and checked against Qwen's chat template.
- **Training crashed at start-up:** `warmup_ratio` was removed in transformers 5. Switched to `warmup_steps` and pinned the library versions.
- **The rendezvous picked `docker0`** instead of `eth0`. Now the IP is resolved with `getent hosts`.
- **The vLLM jobs had start-up and exit problems:** the server needed its environment activated (`ninja` not found), and the batch job never exited after finishing, holding the GPU. Both fixed.
- **Multi-turn requests to the server returned HTTP 400** (the dataset's history format differs from the OpenAI API). Fixed by converting the history in the comparison client.
- **W&B showed the successful run as failed** because of an unclean shutdown. Cosmetic; fixed for later runs.

## Next steps

- Tune hyperparameters one change at a time (e.g. a 3rd epoch, since validation loss was still falling).
- Scale the same scripts to 8-GPU InfiniBand nodes for the full reservation.
