# Command cheat sheet

> 🖥️ **Mac**: run locally · 🔐 **Login**: run on the Slurm login node · Replace `<ID>` with the job ID

---

## Terraform 🖥️

*In `soperator/installations/example`*

```bash
chmod 600 ./soperator-ganguly-sandip-backend-secret.env
umask 077
source ./soperator-ganguly-sandip-backend-secret.env
source .envrc
```

```bash
terraform plan -input=false -out=candidate-soperator.tfplan
terraform show -no-color candidate-soperator.tfplan > soperator-plan.txt
terraform apply candidate-soperator.tfplan
terraform plan          # drift check: expect only 3 routine replacements
```

---

## Login 🖥️

```bash
./login.sh -k ~/.ssh/id_ed25519
ssh -i ~/.ssh/id_ed25519 root@<login-node-ip>
scp -i ~/.ssh/id_ed25519 <file> root@<login-node-ip>:~/validation/
```

---

## Kubernetes 🖥️

```bash
kubectl config use-context nebius-startup-slurm
```

| Check | Command |
|---|---|
| Nodes + labels | `kubectl get nodes -L slurm.nebius.ai/nodeset,topology.nebius.com/tier-1` |
| Slurm cluster status | `kubectl -n soperator get slurmclusters` |
| Soperator pods | `kubectl -n soperator get pods -o wide` |
| Worker pods | `kubectl -n soperator get pods worker-0 worker-1` |
| Stuck volumes | `kubectl get pvc -A \| grep -v Bound` |
| Unhealthy pods | `kubectl get pods -A \| grep -v -E ' Running \| Completed '` |
| Recent warnings | `kubectl get events -A --field-selector type=Warning --sort-by=.lastTimestamp \| tail -15` |
| Worker init log | `kubectl -n soperator logs worker-0 -c worker-init --tail=30` |
| Health checks | `kubectl -n soperator get activechecks` |
| Flux releases | `kubectl -n flux-system get helmreleases` |

---

## Slurm 🔐

| Purpose | Command |
|---|---|
| Partitions / node states | `sinfo` |
| Per node: CPUs, memory | `sinfo -N -l` |
| Node, partition, state, GRES, reason | `sinfo -N -o "%N %P %T %G %E"` |
| Jobs in queue | `squeue` |
| Jobs, custom format | `squeue -u root -o "%.8i %.20j %.8T %.10M %R"` |
| Recent job history | `sacct -X -S now-2hours -o JobID,JobName%25,State,Elapsed,ExitCode` |
| One job's result | `sacct -j <ID> -o JobID,JobName%25,State,Elapsed,ExitCode` |
| Node details | `scontrol show node worker-0` |
| All nodes, key fields | `scontrol show node \| grep -E 'NodeName\|Gres=\|CPUTot\|State=\|Reason='` |
| Job details | `scontrol show job <ID>` |
| Network topology | `scontrol show topology` |
| Controller up? | `scontrol ping` |
| Cancel job | `scancel <ID>` |
| Undrain a node | `scontrol update nodename=worker-0 state=resume` |

**Allocation test** (node-list expansion):

```bash
salloc -N2 --gpus-per-node=1 bash -c 'echo "NODELIST=$SLURM_JOB_NODELIST"; scontrol show hostnames "$SLURM_JOB_NODELIST"'
```

---

## GPU · driver · CUDA · NCCL versions 🔐

**GPUs and driver**

```bash
srun -N2 --ntasks-per-node=1 --gpus-per-node=1 nvidia-smi -L
srun -N2 --ntasks-per-node=1 --gpus-per-node=1 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun -N1 --gpus=1 nvidia-smi | head -5          # driver + max CUDA version
srun -N2 --ntasks-per-node=1 --gpus-per-node=1 bash -c 'echo "$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; ls -l /dev/nvidia0'
```

**System libraries**

```bash
ls -l /usr/lib/x86_64-linux-gnu/libnccl.so.*    # system NCCL (used by nccl-tests)
python3 --version
```

**PyTorch venv** (torch / CUDA runtime / bundled NCCL)

```bash
/opt/venvs/train/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())"
srun -N2 --ntasks-per-node=1 --gpus-per-node=1 /opt/venvs/train/bin/python -c \
  'import torch,socket; print(socket.gethostname(), torch.cuda.is_available(), torch.cuda.get_device_name(0))'
/opt/venvs/train/bin/pip list | grep -i -E 'torch|transformers|trl|peft|numpy'
```

| Component | Current |
|---|---|
| Driver | 580.173.02 (CUDA 13.0) |
| PyTorch | 2.11.0+cu128 (CUDA 12.8) |
| NCCL in PyTorch | 2.28.9 |
| NCCL in system (nccl-tests) | 2.30.7 |
| Python | 3.12.3 |

---

## Network 🔐

```bash
srun -N2 --ntasks-per-node=1 bash -c 'echo $(hostname); ip -4 -br addr'
getent hosts worker-0 worker-1                  # eth0 IPs via DNS
```

> ⚠️ Use **`eth0`**. `docker0` (`169.254.0.1`) is down and identical on every worker, and `hostname -I` lists it first.

---

## NCCL test: 2 nodes over TCP 🔐

*About 2 min · expect busbw ≈ **1.1 GB/s***

```bash
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
srun -N2 --ntasks-per-node=1 --gpus-per-node=1 --mpi=pmix all_reduce_perf_mpi \
  -b 8 -e 1G -f 2 -g 1 -n 20 -w 5 -c 1 2>&1 | tee ~/nccl-2node.log
```

```bash
grep -v "NCCL INFO" ~/nccl-2node.log                                     # table with header
grep -E "Rank|nRanks|Using network|Out of bounds|Avg bus" ~/nccl-2node.log
```

> Use **`*_perf_mpi`**: plain `all_reduce_perf` runs 1 rank per node. Quicker run: `-e 256M`.

---

## Python venv (once) 🔐

*The jail is shared, so installing once on the login node covers both workers*

```bash
mkdir -p /opt/venvs /data/models /data/hf-cache /data/checkpoints
python3 -m venv /opt/venvs/train
/opt/venvs/train/bin/pip install --upgrade pip
/opt/venvs/train/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
/opt/venvs/train/bin/pip install numpy
source /opt/venvs/train/bin/activate            # optional, for short commands
```

---

## DDP smoke test 🔐

```bash
cd ~/validation && sbatch ddp_smoke.sbatch
squeue
tail -f ~/validation/ddp-smoke-<ID>.out
```

```bash
grep -v "NCCL INFO" ~/validation/ddp-smoke-<ID>.out                       # expect SMOKE TEST PASSED
grep -E "NET/(Socket|IB)|Using network" ~/validation/ddp-smoke-<ID>.out | sort -u
sacct -j <ID> -o JobID,JobName,State,Elapsed,ExitCode
```

---

## GPU monitoring (nvidia-smi) 🔐

> The **login node has no GPU**. Run these **on a GPU node**, or from the login node **inside a job** with `srun --jobid <ID> --overlap …` (section B).
> **Top table** of `nvidia-smi` = totals for the whole GPU (all processes). **Bottom table** = per process.

### A. Snapshot on a GPU node

*On the node itself: `ssh worker-0` from the login node, or from your Mac `kubectl -n soperator exec -it worker-0 -c slurmd -- bash`.*

| What | Command |
|---|---|
| Full snapshot (utilisation, memory, power, temperature, processes) | `nvidia-smi` |
| **Utilisation + memory per GPU, every 2 s** | `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv -l 2` |
| Same plus power and temperature | `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 2` |
| Live, 1 line/s (`sm` % = busy, `mem` %, `fb` MB used) | `nvidia-smi dmon -s um` |
| Live, per **process** (which PID uses the GPU) | `nvidia-smi pmon -s um` |
| Memory per process | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` |
| Driver / CUDA version | `nvidia-smi \| head -4` |

Stop the `-l` / `dmon` / `pmon` loops with **Ctrl-C**. Add `-c 20` to `dmon`/`pmon` to stop after 20 samples.

### B. For a specific Slurm job, e.g. job **37** (from the login node)

```bash
squeue -j 37                                              # running? on which node(s)?
scontrol show job 37 | grep -E 'JobState|NodeList|TRES|RunTime'    # state, nodes, GPUs allocated, run time
```

> ⚠️ `srun --jobid <ID>` runs on **all nodes of the job**. For a 2-node job, plain `nvidia-smi` prints both nodes' output **interleaved**, with no node name. Always add **`--label`** (prefixes each line with `0:` = first node, `1:` = second) or pick one node with **`-w`**.

**Snapshot, one line per GPU** (any job; recommended):

```bash
srun --jobid 37 --overlap --label nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
# 0: 0, 100 %, 44237 MiB, 81559 MiB, 606 W     ← worker-0
# 1: 0, 100 %, 40941 MiB, 81559 MiB, 634 W     ← worker-1
```

**Same, refreshed every 2 s** (Ctrl-C to stop):

```bash
srun --jobid 37 --overlap --label nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader -l 2
```

**Live per-second view** (`sm` % = busy, `fb` = memory MB):

```bash
srun --jobid 37 --overlap --label nvidia-smi dmon -s um -c 20
```

**Full `nvidia-smi` table for ONE node:**

```bash
srun --jobid 37 --overlap -N1 -w worker-0 nvidia-smi
srun --jobid 37 --overlap -N1 -w worker-1 nvidia-smi
```

**20-second average per node** (one summary line each; good for the demo):

```bash
srun --jobid 37 --overlap bash -c 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 | head -20 | awk -F", " -v h=$(hostname) "{u+=\$1; m+=\$2; n++} END {printf \"%s: GPU util %.0f%%, memory %.1f GB\n\", h, u/n, m/n/1024}"'
# worker-1: GPU util 100%, memory 40.0 GB
# worker-0: GPU util 99%, memory 43.2 GB
```

`--overlap` joins the running job without needing free resources. The `cpu-bind=MASK …` lines Slurm prints first are harmless (they show which node each task ran on).

### C. Other views

| Where | What |
|---|---|
| Nebius console → Compute → GPU instance → **Monitoring** | Utilisation / memory graphs over time (use for the > 80 % screenshots) |
| W&B run → **System** tab | GPU charts for the rank-0 node (worker-0) of a training run |
| `sstat -j 37.0 --format=JobID,AveCPU,MaxRSS` | CPU and RAM of the job's main step (not GPU) |

---

## Disk / filesystems 🔐

```bash
df -h / /home /mnt/local-data 2>/dev/null
du -sh /opt/venvs/train /data/* 2>/dev/null
```

| Mount | Filesystem | Use for |
|---|---|---|
| `/`, `/root`, `/opt`, `/data` | Jail, 1 TiB shared | Venvs, models, data, checkpoints, scripts |
| `/home` | NFS, 3.6 TiB shared | Normal users' homes (empty in this PoC) |
| `/mnt/local-data` | Per-worker 1 TiB disk | Scratch only, not shared |

---

## Fine-tuning pipeline: step by step 🔐

*Login node (`./login.sh -k ~/.ssh/id_ed25519`). Scripts: `~/eval` (from `poc/eval/`), `~/training` (from `poc/training/`). Run one step at a time; check the result before the next.*

| Step | What | Where | Time |
|---|---|---|---|
| 0 | Check inputs | login | – |
| 1 | vLLM environment | login | ~5 min |
| 2 | Training libraries | login | ~2 min |
| 3 | Download models | login | ~10 min |
| 4 | Quick inference test (20 items) | sbatch, 1 GPU | ~3 min |
| 5 | **BEFORE** benchmark | sbatch, 1 GPU | ~5–10 min |
| 6 | Training pipeline test (1.5B, 30 steps) | sbatch, 2 GPUs | ~5 min |
| 7 | **Training** (7B, 2 epochs) | sbatch, 2 GPUs | ~1–2 h |
| 8 | **AFTER** benchmark | sbatch, 1 GPU | ~5–10 min |
| 9 | Before/after table | login | seconds |
| 10 | **Exercise 2**: server (base + fine-tuned) | sbatch, 1 GPU | until `scancel` |
| 11 | **Exercise 2**: live comparison | login / Mac | – |

### 0. Check inputs

```bash
ls ~/eval ~/training
wc -l /data/datasets/toolace-clean/*.jsonl   # train 9462 · validation 526 · test 526 · eval_inputs 725 · eval_expected 725
sinfo                                        # both workers idle
```

### 1. vLLM environment (once)

```bash
cd ~/eval && bash setup_vllm_env.sh          # last line prints the vllm / torch versions
```

### 2. Training libraries (once)

```bash
cd ~/training && bash setup_train_env.sh     # prints torch / transformers / trl / peft versions
```

### 3. Download models (once)

```bash
cd ~/eval && bash download_models.sh
du -sh /data/models/*                        # Qwen2.5-1.5B-Instruct ~3 GB · Qwen2.5-7B-Instruct ~15 GB
```

### 4. Quick inference test (20 items)

```bash
cd ~/eval && sbatch --export=ALL,RUN=quick,LIMIT=20 eval.sbatch
squeue
cat eval-<ID>.out                            # ends with {"items": 20, "seconds": …}
head -3 /data/eval-runs/quick/predictions.jsonl   # answers should contain <tool_call> … </tool_call>
```

### 5. BEFORE benchmark: base model

```bash
cd ~/eval && sbatch --export=ALL,RUN=base-7b eval.sbatch
cat /data/eval-runs/base-7b/metrics.json     # exact_match_pct · right_tool_pct · correct_no_call_pct · malformed_pct
```

### 6. Training pipeline test: 1.5B, 30 steps

```bash
cd ~/training && sbatch --export=ALL,MODEL=/data/models/Qwen2.5-1.5B-Instruct,RUN=test-1.5b,MAX_STEPS=30 train.sbatch
tail -f train-lora-<ID>.out                  # "trainable params …", loss lines, "adapter saved to …"
ls /data/checkpoints/test-1.5b/adapter       # adapter_model.safetensors, adapter_config.json
```

### 7. Training: 7B, 2 epochs

```bash
# optional, once: live charts in Weights & Biases (key stays on the cluster in /root/.netrc)
/opt/venvs/train/bin/pip install wandb && /opt/venvs/train/bin/wandb login
cd ~/training && sbatch --export=ALL,RUN=qwen7b-lora,WANDB=1 train.sbatch     # or without WANDB=1
tail -f train-lora-<ID>.out                  # loss every 10 steps, eval_loss every 50
srun --jobid <ID> --overlap -N2 --ntasks-per-node=1 nvidia-smi dmon -s u -c 30   # GPU use (target > 80 %)
ls /data/checkpoints/qwen7b-lora/adapter
```

### 8. AFTER benchmark: base + adapter

```bash
cd ~/eval && sbatch --export=ALL,RUN=lora-7b,LORA=/data/checkpoints/qwen7b-lora/adapter eval.sbatch
cat /data/eval-runs/lora-7b/metrics.json
```

### 9. Before/after table

```bash
cd ~/eval && /opt/venvs/vllm/bin/python compare.py /data/eval-runs/base-7b/metrics.json /data/eval-runs/lora-7b/metrics.json
```

### 10. Exercise 2: start the server (serves BOTH models; holds 1 GPU)

```bash
cd ~/eval && sbatch --export=ALL,LORA=/data/checkpoints/qwen7b-lora/adapter serve.sbatch
squeue                                       # node, e.g. worker-0
tail -f vllm-serve-<ID>.out                  # wait for "Application startup complete"
curl http://worker-0:8000/v1/models          # Qwen2.5-7B-Instruct (base) and toolace (fine-tuned)
```

### 11. Exercise 2: compare live

```bash
cd ~/eval && /opt/venvs/vllm/bin/python chat_compare.py --url http://worker-0:8000 --item toolace-02491#1 \
  --inputs /data/datasets/toolace-clean/eval_inputs.jsonl --expected /data/datasets/toolace-clean/eval_expected.jsonl
```

Plain `curl`: change only `"model"` to switch between base and fine-tuned:

```bash
curl -s http://worker-0:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "toolace",
  "messages": [{"role": "user", "content": "What is the weather in Paris and in Tokyo?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "temperature": 0}' | python3 -m json.tool
```

From your Mac: `ssh -i ~/.ssh/id_ed25519 -L 8000:worker-0:8000 root@<login-node-ip>`, then `http://localhost:8000`. Stop: `scancel <ID>`.
