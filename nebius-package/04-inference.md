# 4. Inference: serving the base and fine-tuned models side by side

## Goal

- Serve the **fine-tuned model** on the same Kubernetes cluster.
- Run the **original (untrained) model** too.
- Compare both models on the same requests.

## Design: one vLLM server, two models, one GPU

- **vLLM** (OpenAI-compatible API) runs as a **Slurm job on one worker GPU** inside the same Soperator cluster on Managed Kubernetes.
- It runs with `--enable-lora --lora-modules toolace=<adapter>`, so the **same server answers as both models**, selected by the `model` field of each request:
  - `Qwen2.5-7B-Instruct`: the base model;
  - `toolace`: base + LoRA adapter (fine-tuned).
- `--tool-call-parser hermes` turns Qwen's `<tool_call>{…}</tool_call>` output into standard OpenAI `tool_calls`.
- **Why this design:**
  - Both GPUs are already held by the Slurm workers, so a separate Kubernetes Deployment would need a GPU freed from Slurm.
  - One server keeps only **one copy of the 15 GB base weights**. The adapter adds ~155 MB.
  - Model, adapter and Python environment are read straight from the shared jail filesystem, so nothing has to be copied or downloaded.
- **Trade-offs:**
  - The endpoint is reached through the login node or an SSH tunnel rather than a Kubernetes Service.
  - LoRA adds some serving overhead: ~3,200 vs ~5,300 tokens/s in batch mode.

## Results

**Benchmark** (725 held-out test items, greedy decoding, same prompts for both models):

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| **Exact match** (right tool + all arguments) | 56.4 % | **68.2 %** | **+11.8 pts** |
| Right tool chosen | 87.6 % | 91.1 % | +3.5 pts |
| **Correctly answered without a tool** | 60.3 % | **78.0 %** | **+17.7 pts** |
| Malformed tool call | 0.4 % | 0.4 % | 0 |

**Live side-by-side example** (`toolace-00488#7`, a parallel-call request):

- **User:** "…could you provide the latest carbon dioxide levels? Also, it would be great to simulate an environment with data from multiple research[ers]…" (5 random users)

| | Answer | Result |
|---|---|---|
| Expected | `Carbon Dioxide Endpoint()` + `Random User Copy(num_users=5)` | |
| Base | `Carbon Dioxide Endpoint()` only | ❌ misses the 2nd call |
| Fine-tuned | `Carbon Dioxide Endpoint()` + `Random User Copy(num_users=5)` | ✅ matches |

## How to run

| Step | Command (login node) |
|---|---|
| Start the server (holds 1 GPU) | `cd ~/eval && sbatch --export=ALL,LORA=/data/checkpoints/qwen7b-lora/adapter serve.sbatch` |
| Check it's up | `curl http://worker-0:8000/v1/models` (lists both models) |
| Side-by-side comparison | `/opt/venvs/vllm/bin/python chat_compare.py --url http://worker-0:8000 --item toolace-00488#7 --inputs /data/datasets/toolace-clean/eval_inputs.jsonl --expected /data/datasets/toolace-clean/eval_expected.jsonl` |
| Stop | `scancel <ID>` |

Full steps are in [03-reproduce.md](03-reproduce.md#c-inference-base-vs-fine-tuned).

## Challenges

- **The vLLM server failed at start-up** (`ninja` not found): the Python environment wasn't activated in the job, so vLLM's GPU kernel compiler wasn't on `PATH`. Fixed by activating the environment in `serve.sbatch`.
- **The live comparison got HTTP 400 on multi-turn requests.** The dataset stores earlier tool calls in Hugging Face's format, while the OpenAI API needs call IDs and JSON-string arguments. Fixed by converting the history in `chat_compare.py`.
- **The batch-inference job never exited** after finishing, holding the GPU, because vLLM's background processes kept Python alive. Fixed with an explicit exit in `run_inference.py`.
