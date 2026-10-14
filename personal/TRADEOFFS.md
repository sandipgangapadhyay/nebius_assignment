# Design Decisions & Trade-offs

Each decision made for the Soperator fine-tuning PoC, with the options considered, why this one was chosen, and what it costs. Bugs and fixes are tracked separately in [FINDINGS.md](FINDINGS.md).

**Constraints that drive almost every decision below**

| Constraint | Consequence |
|---|---|
| 2 × H100 80 GB, **one GPU per VM** | No NVLink. Every GPU-to-GPU byte crosses the VM network. |
| **No InfiniBand** (1-GPU preset has no GPU cluster) | NCCL falls back to TCP sockets over Ethernet: much lower bandwidth and higher latency than IB. |
| 16 vCPU / 200 GiB RAM per worker | Limited data-loader CPU; host RAM is fine for a 7–8B model. |
| 1 TiB shared filesystem, 1 TB network disk | Model, dataset and checkpoints must fit comfortably; no room for multi-hundred-GB artifacts. |
| Customer: ML engineers, light infra experience | Prefer standard, well-documented tools over custom or fragile stacks. |
| Target: **> 80 % GPU utilisation** | The design must keep the GPUs computing, not waiting on the network. |

## Decision index

| # | Decision | Choice | Status |
|---|---|---|---|
| D1 | Soperator version | `soperator-v4.1.12-1` tag | ✅ Decided |
| D2 | Worker node-group layout | 2 node groups × 1 node, `max_surge = 0` | ✅ Decided |
| D3 | Interconnect | No GPU cluster; NCCL over TCP | ✅ Forced by hardware |
| D4 | Where to fix the GRES bug | Override in the installation, not the shared module | ✅ Decided |
| D5 | Parallelism strategy | **DDP** (not FSDP / ZeRO-3) | ✅ Decided |
| D6 | Fine-tuning method | **LoRA** (not full fine-tune / QLoRA) | ✅ Decided |
| D7 | Base model | **Qwen2.5-7B-Instruct** for the demo run (3B rejected: non-commercial licence); 1.5B for the smoke test | ✅ Decided |
| D8 | Dataset | **Team-ACE/ToolACE (original)**, converted ourselves (replaced the Qwen-cleaned derivative, see FINDINGS #15) | ✅ Decided |
| D9 | Training framework | Hugging Face TRL + PEFT, launched with `torchrun` via `srun` | ✅ Decided |
| D10 | Storage layout | Models/data/checkpoints/venvs on the jail; code and logs in `/home` (NFS); scratch on `/mnt/local-data` | ✅ Decided |
| D11 | Inference / evaluation / Exercise 2 | vLLM via Slurm on 1 worker GPU: offline batch for the benchmark, OpenAI server (base + LoRA in one) for serving | ✅ Decided; scripts in `poc/eval/` |
| D13 | Data split and evaluation | 90/5/5 → 9,202 / 511 / 511 (seed 42); before/after on the same test set with tool-call metrics | ✅ Split done; metrics decided |
| D14 | How the GPU-fabric check exempts 1-GPU presets | Read the catalogue's `gpu_cluster_compatible` flag (not a hard-coded preset name) | ✅ Decided |
| D12 | How much to change the stock recipe | Minimal change: only what the hardware or instructions require; tune after deploy evidence | ✅ Decided |

---

## D1 — Use the latest stable release tag, not `main`

**Choice:** `soperator-v4.1.12-1` (latest `soperator-v*` tag on the remote; Helm chart `4.1.12`).

| Option | Pros | Cons |
|---|---|---|
| **Release tag** ✅ | Tested together (Terraform + charts + images); reproducible; what a customer would run | Misses unreleased fixes |
| `main` branch | Newest fixes | Moving target; Terraform and charts can be out of sync; not reproducible for the customer |

**Why:** a PoC must be reproducible by the customer weeks later. The assignment also requires it explicitly.

---

## D2 — Two node groups of one node each, zero surge

**Choice:** `size = 2`, `nodes_per_nodegroup = 1`, which creates `worker-0` and `worker-1` with `max_surge = 0` and `max_unavailable = 50 %`.

| Option | Pros | Cons |
|---|---|---|
| **2 groups × 1 node** ✅ | Each GPU node is updated or repaired on its own; no surge, so it never needs a 3rd GPU | Two node groups to manage; required a code change (FINDINGS #2) |
| 1 group × 2 nodes (stock default) | Simpler | An update with surge would need a 3rd GPU (outside quota); without surge, a group update can disrupt both GPUs |

**Why:** quota is exactly 2 GPUs. Any rolling update that adds a node first (surge) would fail or hang on quota.

**Trade-off accepted:** during maintenance one GPU is offline, so a running 2-node job is interrupted. That's acceptable for a PoC, and it's why training checkpoints regularly (D10).

---

## D3 — No GPU cluster / InfiniBand; NCCL over TCP

**Choice:** omit `gpu_cluster`; NCCL uses sockets (`NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=eth0`).

This choice is **forced by the hardware**: the `1gpu-16vcpu-200gb` preset cannot join a GPU cluster.

**Consequence (the key one):** cross-node bandwidth is Ethernet-class rather than 8 × 400 Gb/s IB. Every later decision (D5, D6) minimises the data that crosses the network. Measure it with `iperf3` between the workers and quote the real number in the demo.

**For the 512-GPU production phase:** 8-GPU nodes with an IB GPU cluster. The training code doesn't change; only the NCCL environment and the parallelism config do.

---

## D4 — Fix the GRES bug in the installation, not the shared module

**Choice:** override `gres_config` for the 1-GPU H100 preset in `installations/example/main.tf` (FINDINGS #3).

| Option | Pros | Cons |
|---|---|---|
| **Installation-level override** ✅ | Small, contained change; easy to show and review; no risk to other platforms | Matches on a preset name; not reusable |
| Preset-aware map in `modules/available_resources/gres.tf` | Correct long-term fix; works for every 1-GPU preset | Touches a module shared by all platforms; larger review surface |

**Why:** for a time-boxed PoC the smallest correct change wins. The proper fix (GRES keyed by preset, or generated from GPU count) is the upstream follow-up. Mention it as the "if I had more time" answer.

---

## D5 — DDP, not FSDP

**Choice:** PyTorch DistributedDataParallel: each GPU holds a full model replica, and gradients are all-reduced every step.

### How they differ

| | **DDP** | **FSDP (FULL_SHARD) / ZeRO-3** |
|---|---|---|
| What each GPU holds | Full model, full gradients, full optimizer state | 1/N of the parameters, gradients and optimizer state |
| Communication per step | **1 all-reduce of gradients** | All-gather of parameters (forward) + all-gather (backward) + reduce-scatter of gradients |
| Bytes on the wire per GPU (N = 2, ring) | ≈ 1 × gradient size | ≈ 1.5 × parameter size (three collectives × ½ each) |
| Overlap | Gradient buckets all-reduce during the backward pass | Prefetch helps, but parameter all-gathers sit on the critical path of every layer |
| Solves | Throughput scaling | **Memory**: models whose training state does not fit on one GPU |

### The numbers for this setup (Qwen2.5-7B, ~7.6 B parameters, bf16)

| Scenario | Memory per GPU | Network per step per GPU | Verdict |
|---|---|---|---|
| Full fine-tune + DDP | ~16 bytes/param × 7.6 B ≈ **122 GB** | ~15 GB | ❌ Does not fit in 80 GB |
| Full fine-tune + FSDP | ≈ 61 GB + activations | **~23 GB** | ⚠️ Fits barely; at ~1–3 GB/s Ethernet that is **8–20 s of communication** against ~1–3 s of compute, so GPU use is likely below 30 % |
| **LoRA + DDP** ✅ | ~15 GB frozen weights + small adapter state + activations | **~80–160 MB** (adapter gradients only) | ✅ Lots of room; communication well under 0.2 s and overlapped with backward |

### Measured on the PoC cluster (nccl-tests `all_reduce_perf_mpi`, 2 nodes × 1 H100, NCCL over TCP)

Inter-node all-reduce **busbw ≈ 1.1 GB/s (≈ 9 Gbit/s)**, latency ≈ 0.1 ms. Time to synchronise once per step:

| What crosses the network per step | Size | Time at 1.1 GB/s |
|---|---|---|
| **LoRA adapter gradients (DDP), our choice** | ~80–160 MB | **~0.1–0.15 s**, overlapped with backward |
| Full fine-tune gradients, 7B (DDP) | ~15 GB | ~14 s |
| Full fine-tune, 7B (FSDP: 2× all-gather + reduce-scatter) | ~23 GB | ~21 s |

With compute at roughly 1–3 s per step, full fine-tuning would leave the GPUs idle more than 80 % of the time; LoRA + DDP leaves the network almost unnoticed.

### Why DDP

1. **FSDP exists to save memory.** With LoRA (D6) the model already fits on one GPU with large headroom, so sharding buys nothing.
2. **FSDP costs bandwidth, and bandwidth is the scarcest resource here** (D3). It would move about 1.5× more data than full-gradient DDP and about 150× more than LoRA DDP.
3. **One GPU per node means FSDP has no fast local fabric to use.** On 8-GPU nodes, HYBRID_SHARD shards over NVLink inside the node and replicates across nodes. Here every shard boundary is a network hop.
4. **Simplest to operate and debug** for an ML team with limited infrastructure experience: no sharding policy, no wrap policy, and checkpoints are ordinary state dicts.

### When I would switch to FSDP

- Full fine-tuning, or models above about 13B, where training state no longer fits on one GPU.
- QLoRA on 70B-class models (4-bit weights plus activations get tight on 80 GB).
- The production phase on 8-GPU IB nodes, using **HYBRID_SHARD**: shard inside the node over NVLink, replicate across nodes over IB.

---

## D6 — LoRA, not full fine-tuning or QLoRA

**Choice:** LoRA adapters (rank 16–64) on all linear layers; base model frozen in bf16.

| Option | Memory | Network (with DDP) | Quality for function calling | Notes |
|---|---|---|---|---|
| Full fine-tune | ~122 GB (needs FSDP) | ~15–23 GB/step | Highest ceiling | Too heavy for 2 × 1-GPU over Ethernet (see D5) |
| **LoRA (bf16)** ✅ | ~15 GB + activations | ~80–160 MB/step | Very close to full fine-tune for format/behaviour tasks such as tool calling | Small adapter checkpoints (~100–300 MB) |
| QLoRA (4-bit) | ~5 GB + activations | Same as LoRA | Slightly lower; dequantisation overhead | Only worth it for much larger models; slower per step, which wastes GPU throughput we have spare memory for |

**Why:** function calling teaches a *format and behaviour* (emit `<tool_call>{json}</tool_call>`, pick the right tool, fill arguments), not new world knowledge. LoRA is the standard, well-proven way to do that. It also makes D5 (DDP) cheap on the network and keeps checkpoints tiny on the 1 TiB shared filesystem.

**Trade-off accepted:** a lower quality ceiling than full fine-tuning, which doesn't matter for this task.

---

## D7 — Base model: Qwen2.5-7B-Instruct (3B considered and rejected)

**Choice:** [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) for the demo run. Use `Qwen2.5-1.5B-Instruct` for the first smoke test of the pipeline. The model is a single parameter in the training script, and the dataset (D8) works with every Qwen2.5 size.

| Candidate | Licence (checked on HF) | Fit on 1 × H100 (LoRA) | GPU-utilisation outlook | Verdict |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | Apache-2.0 | Trivial | Low; useful only for testing the pipeline | ✅ Smoke test |
| Qwen2.5-3B-Instruct | **Qwen RESEARCH licence: "FOR NON-COMMERCIAL PURPOSES ONLY"** | Trivial | At risk: small kernels; one epoch (~10 k samples) lasts only minutes, too short to show on dashboards | ❌ Licence blocks commercial use by the customer |
| **Qwen2.5-7B-Instruct** ✅ | **Apache-2.0**, ungated | ~15 GB weights, lots of headroom | Good: enough compute per step to keep the GPU busy; epochs long enough to show sustained > 80 % | Best balance |
| Qwen2.5-14B-Instruct | Apache-2.0 | Fits with LoRA (~30 GB) | Excellent | Stretch option if time allows |
| Qwen3-8B | Apache-2.0 | Fits | Good | Hybrid thinking template; the dataset is built for the Qwen2.5 template |
| Llama-3.1-8B-Instruct | Llama licence, **gated** | Fits | Good | Gating friction; different tool format |

**Why not 3B (the reason that would surprise an interviewer):** Qwen2.5 sizes do **not** all share a licence. 0.5B, 1.5B, 7B, 14B and 32B are Apache-2.0. **3B and 72B are under the Qwen Research Licence (non-commercial only).** The customer is a VC-funded startup building a product: a PoC built on a model they cannot ship turns into a redo later.

**Why 7B:**
1. **Commercial-friendly** (Apache-2.0) and ungated, so the customer can reproduce and ship it.
2. **Utilisation target:** larger matrix multiplies keep the H100 busy. With 3B, the step time gets so short that data loading and launch overhead show up, and the job ends before the dashboards show a sustained level.
3. **Credibility:** the customer plans a 512-GPU reservation; 7B is a realistic production-sized base for an agent model.
4. Still fits easily under LoRA + DDP (D5/D6), so DDP stays the right choice.

**Trade-off accepted:** each 7B run takes roughly 2× as long as 3B. The 1.5B smoke test keeps the debugging loop fast.

---

## D8 (revised) — Dataset: original Team-ACE/ToolACE, converted ourselves

**Change of decision:** the earlier choice, `tryumanshow/ToolACE-Qwen-cleaned`, was **replaced** after label review (FINDINGS #15): its conversion dropped object/array arguments, parallel calls, exact tool names and `required`/`enum` schemas, leaving ~1/3 of sampled labels missing values the user had stated. The **original** `Team-ACE/ToolACE` (Apache-2.0, 11,300 rows) has complete labels.

| | Original ToolACE + own conversion ✅ | Qwen-cleaned derivative |
|---|---|---|
| Label completeness (25-sample manual review) | 25 / 25 | ~16 / 25 |
| Parallel calls | 4,125 turns kept | collapsed to 1 call |
| Schemas | full (`required`, `enum`, nested objects) | flattened |
| Licence | Apache-2.0 | none stated |
| Effort | own parser + validator (`poc/data/`) | "ready to use" |

**Why it's worth the effort:** garbage in, garbage out. The model would faithfully learn to drop arguments. Every label is now schema-validated (0 violations / 18,161 calls) and the conversion is reproducible (`poc/data/README.md`). The conversion targets Qwen's native format with OpenAI-style tool schemas, keeping train and serve prompts identical.

---

## D8 (superseded) — Dataset: tryumanshow/ToolACE-Qwen-cleaned (with a 3-line format fix)

**Choice:** [`tryumanshow/ToolACE-Qwen-cleaned`](https://huggingface.co/datasets/tryumanshow/ToolACE-Qwen-cleaned): 10,547 conversations, ungated. It is derived from [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) (Apache-2.0).

**Format (checked against the HF dataset viewer):** two columns, each a JSON string:
- `tools`: list of `{name, description, parameters}`
- `conversations`: OpenAI-style messages: `user`, `assistant` with `tool_calls`, `tool`, `assistant`

**Why it beats Hermes (the previous choice):**

| | **ToolACE-Qwen-cleaned** ✅ | hermes-function-calling-v1 |
|---|---|---|
| Size | 10.5 k | ~7.1 k |
| Structure | Real `tools` + `tool_calls` fields | Tools baked into a custom system-prompt string |
| Prompt at train time | **Generated by Qwen's own chat template**, the same text vLLM/Transformers produce at inference | Hermes' own system prompt, which differs from what Qwen's template generates when serving |
| Train/serve consistency | High | Mismatch in the system prompt |
| Licence | Not stated on the derived dataset (upstream ToolACE is Apache-2.0) | Apache-2.0 |

The key argument is **train/serve consistency**: because examples are rendered with `tokenizer.apply_chat_template(conversations, tools=tools)`, the model is fine-tuned on exactly the prompt it will see when served with `tools=[...]`.

### ⚠️ "No preprocessing needed" is not quite true

I rendered a real row through Qwen2.5's actual chat template (see FINDINGS #8). There are two formatting defects:

1. **Double-encoded arguments.** `arguments` is stored as a JSON *string*, and the template applies `| tojson` to it, which produces:
   ```
   {"name": "Quotes by Keywords", "arguments": "{\"word\": \"inspiration\"}"}   ← string, wrong
   {"name": "Quotes by Keywords", "arguments": {"word": "inspiration"}}         ← object, correct
   ```
   The model would learn to emit arguments as a quoted string, which breaks vLLM's tool-call parser and every client that expects an object.
2. **Tool responses rendered as a Python repr** (`[{'name': ...}]`, single quotes) because `content` is a list, not a string.

**Fix (applied in the formatting function before `apply_chat_template`):**

```python
for m in conversations:
    for tc in m.get("tool_calls") or []:
        if isinstance(tc["function"]["arguments"], str):
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    if m["role"] == "tool" and not isinstance(m["content"], str):
        m["content"] = json.dumps(m["content"], ensure_ascii=False)
```

**Smaller caveats (documented, not fixed):**
- The "cleaned" tool schemas are simplified: `parameters` is a flat map with `type` mostly `"string"`, and enums from the original ToolACE (for example `trend_type`) are dropped. The model sees less type information than real OpenAI-style JSON Schemas give it.
- No licence on the derived dataset. For production, re-derive it from upstream ToolACE (Apache-2.0) with the same transformation.

**Alternatives considered:** Hermes v1 (fallback; see table), xLAM-60k (gated; needs conversion), APIGen-MT (CC-BY-NC, not usable commercially), raw ToolACE (custom call syntax).

**Interview line:** *"I picked a dataset already aligned to Qwen's native tool-calling template so the training prompt matches the serving prompt. I didn't trust 'no preprocessing needed'; I rendered a sample through the real template and found double-encoded arguments, which I fixed in three lines before training."*

---

## D9 — Framework: Hugging Face TRL + PEFT with `torchrun` under Slurm

**Choice:** TRL `SFTTrainer` + PEFT `LoraConfig`, bf16, FlashAttention-2, sequence packing; launched as `srun torchrun --nnodes=2 --nproc_per_node=1`.

| Option | Pros | Cons |
|---|---|---|
| **TRL + PEFT + torchrun** ✅ | Standard, widely documented; the customer's ML engineers likely already know it; DDP by default; a short, readable script | Less "batteries included" than config-driven tools |
| Axolotl / LLaMA-Factory | YAML-driven, fast to start | Another abstraction to debug; hides what is happening, which is bad for a teaching PoC |
| NVIDIA NeMo / Megatron | Best at very large scale (TP/PP) | Heavy, steep learning curve; overkill for 2 GPUs + LoRA |
| DeepSpeed ZeRO | Flexible memory optimisation | Like FSDP: unnecessary sharding and network cost here (D5) |

**Why:** the customer has to understand and change the example themselves. A short TRL script under `srun` maps directly onto how Slurm is used on the 512-GPU cluster.

**GPU-utilisation levers (> 80 % target):** sequence packing (no padding waste), the largest micro-batch that fits (memory is spare), FlashAttention-2, enough data-loader workers (16 vCPUs available), and dataset/model pre-downloaded to the shared filesystem so nothing is fetched during training.

---

### Final training settings (`poc/training/train.py`)

Kept deliberately minimal and commented for the customer: TRL `SFTTrainer` + PEFT LoRA (r 16, alpha 32, dropout 0.05, all linear layers), 2 epochs, 8 × 2 accumulation × 2 GPUs = 32 conversations per update, lr 1e-4 cosine, 3 % warm-up, bf16, gradient checkpointing, max 4,096 tokens, `eval_loss` every 50 steps. Each conversation is rendered with Qwen's chat template into one training text.

**Simplifications accepted:** loss on the whole text (not only assistant turns; Qwen2.5's template has no generation markers for assistant-only masking), no sequence packing (would need FlashAttention-2 to avoid cross-contamination). Both keep the script short and dependency-light; possible later optimisations if utilisation or quality needs it.

---

## D10 — Storage layout (stock storage, per D12)

The jail is **one** 1 TiB shared filesystem attached to every node: controller, login, system, accounting and both GPU workers. It is `/` inside Slurm. With the stock config, `/home` is a separate NFS share (3.7 TiB disk served by the NFS pod), not part of the jail.

| Data | Location | Why |
|---|---|---|
| Model weights, HF cache (`HF_HOME`), dataset | Jail: `/data/models`, `/data/hf-cache` | Shared filesystem built for large parallel reads; both workers load at the same time; downloaded once from the login node |
| LoRA checkpoints, final adapter, eval results | Jail: `/data/checkpoints/<run>` | Survives node replacement; served directly by vLLM (D11) |
| Python environments (training, vLLM) | Jail: `/opt/venvs` | Identical on every node |
| Code, sbatch scripts, Slurm logs | User's home: as `root` that is `/root` (**jail**); a normal user's would be `/home/<user>` (NFS) | Small files; normal login-node workflow. In this PoC everything runs as root, so NFS `/home` stays empty |
| Scratch / temporary | `/mnt/local-data` (per worker, not shared) | Keeps heavy temporary I/O off shared storage; never put checkpoints here |
| Container images | `/mnt/image-storage` (per worker) | Enroot/Docker cache only, if containers are used |

**`/data` is a directory, not a filesystem.** `/data/...` and `/opt/venvs` are plain folders on the jail root, created once as root on a login node (`mkdir -p /data/{models,hf-cache,checkpoints} /opt/venvs`). The stock tfvars' separate `filestore_jail_submounts` "data" filesystem (`/mnt/data`) is deliberately **not** created, because the assignment provides only one 1 TB shared filesystem, and that is the jail.

**Why not keep models in `/home`?** NFS is served by a single pod on one CPU node, so two workers each reading 15 GB at once would bottleneck there. The jail is a managed shared filesystem attached directly to each VM.

Expected jail usage: about 60–80 GB of the 1 TiB, including the optional merged model for inference.

**Guideline "avoid using the same shared filesystem for 2 different jails":** met. `filestore_jail` uses `spec` (creates a new dedicated filesystem, `soperator-startup-jail`), not `existing = { id = ... }`. No `filestore_jail_submounts` are set, and `nebius compute filesystem list` showed the project had **no filesystems** before the apply (checked 2026-10-07). Keep it that way: never point another cluster's `filestore_jail.existing.id` at this filesystem.

**Open trade-off:** the stock disk sizes (NFS 3.7 TiB, 2 × 1 TiB local-data, 2 × 930 GiB image disk) exceed the 1 TB network-disk allocation. Sizing is still to be decided.

---

## D11 — Inference (Exercise 2): vLLM as a Slurm job, base + LoRA from one server (PROPOSED)

**Requirement:** serve the fine-tuned model on the same Kubernetes cluster, run the original model too, and compare them.

**Constraint:** quota is exactly 2 GPUs, and both are held by Soperator worker pods (`nvidia.com/gpu: 1` each). A separate Kubernetes inference deployment cannot get a GPU unless a Slurm worker is scaled down.

| Option | Pros | Cons |
|---|---|---|
| **vLLM as a Slurm job on a worker** ✅ | No infrastructure change; runs once training releases the GPUs; reads model + adapter straight from the jail; still "on the same k8s cluster" (Soperator runs on mk8s) | Endpoint reached through the login node / SSH tunnel rather than a k8s Service |
| Kubernetes-native vLLM Deployment + Service | Looks like production serving; k8s Service endpoint | Needs a free GPU: scale the Slurm worker NodeSet down to 1, or add a GPU node group (no quota). Model must be reachable from the pod (mount `jail-pvc` or copy it in). More moving parts |
| Transformers `generate()` script | Simplest | Not a serving system; slow; weak demo |

**Comparing base vs fine-tuned with no extra GPU:** start vLLM once with `--enable-lora --lora-modules toolace=<adapter path>`. The same server answers as `Qwen/Qwen2.5-7B-Instruct` (base) and as `toolace` (fine-tuned), selected by the `model` field in each request. Both are served from **one GPU** and one copy of the base weights. The second GPU can run a second replica or the evaluation job.

**Evaluation:** run the same ~5 % held-out ToolACE split through both models with `--tool-call-parser hermes` and `tools=[...]`. Score: valid JSON tool call, correct function name, exact-match arguments, plus correct "no tool needed" answers. Present a before/after table plus a few side-by-side examples.

**Storage impact:** none. Base weights (~15 GB), adapter (~0.1–0.3 GB), optional merged model (~15 GB) and a separate vLLM Python environment (~10 GB) all sit comfortably on the 1 TiB jail. If vLLM runs in a container instead (Enroot/Pyxis or a k8s pod), the ~20 GB image lands on the 512 GiB worker boot disk. **Neither needs the 930 GiB image disk or the NFS disk**, so the FINDINGS #4 recommendations stand.

---

## D13 — Data split and evaluation design

**Split** (fixed seed 42, **group-aware** by first user message so near-duplicates can't leak; 10,514 rows; `poc/data/`):

| Split | Share | ~Rows | Used for |
|---|---|---|---|
| Train | 90 % | 9,462 | LoRA fine-tuning |
| Validation | 5 % | 526 | `eval_loss` during training: spot overfitting, pick the checkpoint |
| Test | 5 % | 526 (725 assistant turns, 919 tool calls) | **Never seen in training.** Before/after comparison, base model vs fine-tuned |

*Why 90/5/5 rather than 80/10/10:* the dataset is small (10.5 k), and every training example helps more than extra test rows. ~500 test conversations (~900 tool-call turns) is already enough for stable percentages (±2–3 points).

**Ground truth:** `poc/data/processed/eval_inputs.jsonl` (prompts, read by inference) + `eval_expected.jsonl` (answers, read only by the scorer), joined on `item_id` and built by `build_eval_set.py`. Answers are kept in a separate file so the model can never see them: each of the **725 assistant turns** of the 526 test conversations becomes an item with *input* (tools + conversation before the turn) and *expected* (the reference answer): **493 tool-call items (919 calls, 199 parallel), 232 text items**. Scoring must handle parallel calls (compare the *set* of calls). The labels come from the dataset; they were never used in training.

**Metrics** (generation-based, greedy decoding, identical prompts for both models). For each assistant turn in a test conversation, give the model the conversation so far plus the `tools` list and compare its reply with the reference:

| Metric | Question it answers |
|---|---|
| Call / no-call accuracy | Did it correctly decide *whether* to call a tool? (~10 % of first turns should *not* call, for example asking for a missing parameter) |
| Format validity | Is the output a parseable `<tool_call>{json}</tool_call>`? |
| Function-name accuracy | Right tool chosen? |
| Arguments exact match | All arguments exactly right (JSON equality)? |
| Argument F1 | Partial credit per key/value |
| Full-call exact match | Name and arguments both correct: the headline number |

Plus validation loss during training, and a handful of side-by-side qualitative examples for the demo.

### Baseline (BEFORE), measured 2026-10-07 (job 38, `/data/eval-runs/base-7b/`)

| Metric | Base Qwen2.5-7B-Instruct |
|---|---|
| **Exact match** (493 tool-call items) | **56.4 %** (single-call 67.3 %, parallel 40.2 %) |
| Right tool chosen | 87.6 % |
| Correctly answered without a tool (232 text items) | 60.3 % |
| Malformed tool call | 0.4 % |
| Throughput | 725 answers in 14.7 s, ~5,300 output tokens/s on 1 H100 |

Where the base model fails (tool-call items): right tool but argument values differ 31.2 % (mostly **adding unrequested optional arguments**, e.g. `outputFormat: "JSON"`, 120 cases, and **different values**, e.g. `locale "en_US"` vs `"en_INT"`, 142), **wrong number of parallel calls** 8.1 %, wrong tool 2.6 %. On text items it often **calls another tool after answering** (92 / 232). These are exactly the behaviours fine-tuning on ToolACE should correct.

### AFTER (fine-tuned), measured 2026-10-07 (job 47, `/data/eval-runs/lora-7b/`, adapter `/data/checkpoints/qwen7b-lora/adapter`)

| Metric | Before (base) | After (fine-tuned) | Change |
|---|---|---|---|
| **Exact match** (493 tool-call items) | 56.4 % | **68.2 %** | **+11.8 pts** |
| … single-call items (294) | 67.3 % | **79.9 %** | +12.6 pts |
| … parallel-call items (199) | 40.2 % | **50.8 %** | +10.6 pts |
| Right tool chosen | 87.6 % | 91.1 % | +3.5 pts |
| **Correctly answered without a tool** (232 text items) | 60.3 % | **78.0 %** | **+17.7 pts** |
| Malformed tool call | 0.4 % | 0.4 % | 0 |
| Throughput (1 H100) | 5,278 tok/s | 3,175 tok/s | LoRA adapter overhead in vLLM |

Item level: 78 tool-call items fixed (wrong → exact) vs 20 regressed (net +58); 47 text items fixed vs 6 regressed. Calls with **unrequested extra arguments fell 120 → 31**; "wrong number of parallel calls" 40 → 26; "argument values differ" 154 → 113. Regressions are mostly label-style mismatches (for example lower-cased values `"living room"` vs `"Living Room"`, or a context-based guess at a search term), which an exact-string metric counts as wrong.

**Same script, same test set, same decoding for both models.** Base model = `Qwen2.5-7B-Instruct`; fine-tuned = base + LoRA adapter. vLLM serves both from one server (D11).

---

## D12 — Minimal change to the stock recipe; tune only on evidence

**Choice:** change the recipe only where (a) the assigned hardware makes the stock config impossible (Findings 1–3), or (b) the assignment or setup package explicitly says so (backend, `public_o11y_enabled`, `backups_enabled`, no `gpu_cluster`). Everything else stays at the stock default for the first deploy, including NFS, disk sizes, shared memory and CPU node sizes.

| Option | Pros | Cons |
|---|---|---|
| **Minimal change, deploy, react to evidence** ✅ | Small diff that's easy to defend; every change is backed by an observed failure; follows the assignment's "investigate the earliest failed startup… document your diagnosis" | Possible failed first attempt (for example PVCs `Pending` on quota); stock sizes are larger than the stated 1 TB network disk |
| Pre-emptively right-size everything | Matches the stated capacity on paper; lower cost | Speculative changes; the NFS one alone surfaced a new bug (Finding 9); bigger diff to explain; risk of over-engineering |

**Risk control:** ask the CSA up front whether the stock disk footprint (3.7 TiB NFS, 2 × 930 GiB image, 2 × 1 TiB local-data) is OK. The right-sizing change is already tested (FINDINGS #4), so it can be applied within minutes if needed.

**Interview line:** *"I separated required adaptations from optimisations. I made the ones the hardware forced, deployed, and let evidence drive the rest. The one thing that didn't match the brief, disk footprint, I raised with the CSA instead of guessing."*

---

## D14 — GPU-fabric check: read `gpu_cluster_compatible`, don't hard-code the preset

**Choice:** the precondition in `modules/k8s/gpu_fabric_validation.tf` requires a `gpu_cluster` only when `is_gpu && gpu_cluster_compatible`, both read from the preset catalogue (FINDINGS #1).

| Option | Pros | Cons |
|---|---|---|
| Hard-code `preset == "1gpu-16vcpu-200gb"` (first version) | Smallest diff | Misses the other 1-GPU presets (`1gpu-20vcpu-224gb`, `1gpu-24vcpu-346gb`); wrong error message; code edit for every new preset |
| Set `gpus = 0` for the preset in the catalogue | No code change | Wrong data: the workers stop being treated as GPU nodes (labels, scheduling) |
| **Read `gpu_cluster_compatible`** ✅ | Flag already exists and is already correct; same rule the node-group code uses (`k8s_ng_workers_v2.tf`); covers every preset; new presets need only a catalogue entry | Slightly larger diff (one new field + two conditions) |

**Why:** the catalogue is the single source of truth for what a preset can do. The bug was that this one check ignored it, so the fix is to read it, not to add a second list. Unlike D4, this touches a shared module, but only to make it consistent with itself; behaviour for 4/8-GPU presets is unchanged.

---

## Likely interview questions — short answers

| Question | Answer |
|---|---|
| Why DDP over FSDP? | *Measured:* 2-node all-reduce here is ~1.1 GB/s over TCP. A full 7B fine-tune would spend ~14–21 s per step just syncing; LoRA's ~100 MB of gradients take ~0.1 s. FSDP saves memory at the cost of ~1.5× more traffic every step. With LoRA the model already fits on one GPU, and the network (Ethernet, no IB, 1 GPU/node) is the bottleneck, so DDP with ~100 MB of adapter gradients per step keeps GPUs above 80 %. |
| When would you use FSDP? | Full fine-tunes or models above ~13B, or on 8-GPU IB nodes using HYBRID_SHARD (shard over NVLink, replicate over IB). |
| Why LoRA? | Function calling is format/behaviour learning, where LoRA matches full fine-tuning closely, at ~1 % of trainable parameters, a tiny checkpoint and negligible network traffic. |
| Why not QLoRA? | We have spare memory; 4-bit dequantisation slows each step and lowers quality. QLoRA is for fitting much larger models. |
| Why Qwen2.5-7B and not 3B? | Qwen2.5-3B is under the Qwen Research Licence (non-commercial only), so the customer couldn't ship it. 7B is Apache-2.0, fits 1 × H100 with LoRA easily, and does enough work per step to sustain > 80 % utilisation. |
| Why this dataset? | ToolACE-Qwen-cleaned is rendered through Qwen's own chat template, so the training prompt equals the serving prompt. It has 10.5 k samples with real `tools` / `tool_calls` structure. |
| Did it really need no preprocessing? | No. Rendering a sample through the real template showed double-encoded `arguments` and Python-repr tool responses; a 3-line fix before `apply_chat_template` corrects both (FINDINGS #8). |
| What does no InfiniBand change? | NCCL runs over TCP, so every design choice minimises bytes over the wire; health checks such as `ib-gpu-perf` are expected to flag nodes. |
| How does this scale to 512 H100s? | Same script; switch to 8-GPU IB nodes with a GPU cluster. For bigger models or full fine-tunes switch DDP to FSDP HYBRID_SHARD and raise the global batch. |
| How do you compare base and fine-tuned without a third GPU? | One vLLM server with `--enable-lora` serves both the base model and the LoRA adapter; the request's `model` field chooses which. One GPU, one copy of the base weights. |
| Why not just hard-code the 1-GPU preset in the fabric check? | The catalogue already says `gpu_cluster_compatible = false` for every 1-GPU preset and the node-group code uses it; the check was the one place ignoring it. Reading the flag covers all 1-GPU presets and needs no code change for new ones (D14). |
| Why 2 node groups of 1 node? | Quota is exactly 2 GPUs; with surge 0 and one node per group, updates never need a 3rd GPU. |

## Change log

| Date | Update |
|---|---|
| 2026-10-12 | D14: GPU-fabric precondition switched from a hard-coded preset name to the catalogue's `gpu_cluster_compatible` flag (FINDINGS #1 revised). |
| 2026-10-07 | Initial: D1–D9 decided, D10 proposed. Model and dataset checked against the HF API (licence, gating, column format). |
| 2026-10-07 | D10 finalised for stock storage: big shared files on the jail, small working files on NFS `/home`. |
| 2026-10-07 | D11 decided: `poc/eval/` (vLLM batch benchmark + `serve.sbatch` OpenAI server with base + LoRA; scorer tested with synthetic predictions). |
| 2026-10-07 | D13: ground truth split into `eval_inputs.jsonl` (model) and `eval_expected.jsonl` (scorer only), to rule out answer leakage. |
| 2026-10-07 | **D8 revised:** switched to original Team-ACE/ToolACE with own conversion (FINDINGS #15); D13 numbers updated (9,462/526/526; 725 eval items). |
| 2026-10-07 | D13: dataset cleaned and split (9,202 / 511 / 511); D8 tool schemas converted to OpenAI function format for train/serve consistency. |
| 2026-10-07 | D5: added measured inter-node bandwidth (nccl-tests: ~1.1 GB/s busbw over TCP) and per-step sync times. |
| 2026-10-07 | D12: minimal-change principle; NFS disable reverted, stock storage kept for the first deploy. |
| 2026-10-07 | D11 proposed: Exercise 2 inference via vLLM Slurm job, base + LoRA from one server; storage plan unchanged. |
| 2026-10-07 | D7 confirmed: Qwen2.5-7B-Instruct for the demo run, 1.5B for the smoke test. |
| 2026-10-07 | D8 changed from Hermes v1 to ToolACE-Qwen-cleaned (train/serve template consistency, more data) with a format fix. D7: Qwen2.5-3B evaluated and rejected (non-commercial licence); 7B kept, 1.5B added for smoke tests. |
