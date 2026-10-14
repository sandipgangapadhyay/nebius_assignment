# 3. How to deploy and reproduce

> 🖥️ = run on your machine · 🔐 = run on the Slurm login node · `<ID>` = Slurm job ID

## Prerequisites

- **Tools on your machine:** Terraform, `kubectl`, the Nebius CLI (logged in), `jq`, `yq`, and an SSH key pair.
- **Remote state:** the provided S3 backend credentials file.
- **Optional, Weights & Biases API key** for live training charts. Get it from [wandb.ai/authorize](https://wandb.ai/authorize), then on the login node, once:
  `/opt/venvs/train/bin/wandb login` (it prompts for the key and stores it in `/root/.netrc` on the shared filesystem, so both workers can use it).

## A. Infrastructure 🖥️

**1. Configure.** `soperator/installations/example/terraform.tfvars` is already set for this PoC (2 × 1-GPU workers, 1 TiB jail). Optionally replace the SSH public key with your own.

**2. Load credentials and the remote-state backend.** First set `TF_BACKEND_AWS_ACCESS_KEY_ID` and `TF_BACKEND_AWS_SECRET_ACCESS_KEY` in `soperator-ganguly-sandip-backend-secret.env`; they are masked in this package.

```bash
cd soperator/installations/example
source ./soperator-ganguly-sandip-backend-secret.env   # provided S3 backend credentials
source .envrc                      # Nebius IAM token, subnet, backend override
```

**3. Deploy.** The apply takes about 50 minutes.

```bash
terraform init
terraform validate
terraform plan -input=false -out=soperator.tfplan
terraform apply soperator.tfplan
terraform plan                     # drift check: expect no real changes
```

**4. Check cluster health.**

```bash
kubectl config use-context nebius-startup-slurm
kubectl -n soperator get slurmclusters                         # Available
kubectl -n soperator get pods worker-0 worker-1 login-0        # Running
kubectl get pods -A | grep -v -E ' Running | Completed '       # nothing unhealthy
kubectl get pvc -A | grep -v Bound                             # no stuck volumes
```

**5. Log in and check Slurm** 🔐

```bash
./login.sh -k ~/.ssh/id_ed25519    # or: ssh -i ~/.ssh/id_ed25519 root@<login-node-ip>
sinfo -N -o "%N %T %G"             # worker-0/1 idle, gpu:nvidia_h100_80gb_hbm3:1
```

**6. Run the 2-node NCCL test** 🔐 Expect 0 errors and a bus bandwidth of about 1.1 GB/s.

```bash
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
srun -N2 --ntasks-per-node=1 --gpus-per-node=1 --mpi=pmix \
  all_reduce_perf_mpi -b 8 -e 1G -f 2 2>&1 | tee ~/nccl-2node.log
```

## B. Training and evaluation

**1. Build the dataset** 🖥️ Downloads ToolACE, then converts, validates and splits it.

```bash
cd soperator/poc/data
REV=6bda777c88d21e5a204703c1ee45597a8fa4f734
curl -sSL -o raw/toolace-original-data.json "https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/$REV/data.json"
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python clean_dataset.py && .venv/bin/python build_eval_set.py
```

**2. Upload the scripts and data** 🖥️

```bash
scp -r -i ~/.ssh/id_ed25519 ../eval ../training root@<login-node-ip>:~/
ssh -i ~/.ssh/id_ed25519 root@<login-node-ip> mkdir -p /data/datasets/toolace-clean
scp -i ~/.ssh/id_ed25519 processed/*.jsonl root@<login-node-ip>:/data/datasets/toolace-clean/
```

**3. One-time setup** 🔐 Python environments and models. The jail filesystem is shared, so installing once covers both workers.

```bash
cd ~/eval && bash setup_vllm_env.sh            # /opt/venvs/vllm
cd ~/training && bash setup_train_env.sh       # /opt/venvs/train
cd ~/eval && bash download_models.sh           # /data/models/Qwen2.5-{1.5B,7B}-Instruct
```

**4. Before: benchmark the base model** 🔐 (1 GPU, ~5 min)

```bash
cd ~/eval && sbatch --export=ALL,RUN=base-7b eval.sbatch
```

**5. Optional: pipeline test** 🔐 (1.5B, 30 steps, 2 GPUs, ~3 min)

```bash
cd ~/training && sbatch --export=ALL,MODEL=/data/models/Qwen2.5-1.5B-Instruct,RUN=test-1.5b,MAX_STEPS=30 train.sbatch
```

**6. Fine-tune 7B with LoRA on 2 nodes** 🔐 (2 GPUs, ~45 min)

```bash
cd ~/training && sbatch --export=ALL,RUN=qwen7b-lora train.sbatch     # add WANDB=1 for W&B charts
tail -f train-lora-<ID>.out                                           # loss, eval_loss
srun --jobid <ID> --overlap --label nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

- Checkpoints are saved every 100 steps. Resubmitting the same `RUN` resumes from the last one.
- The adapter is written to `/data/checkpoints/qwen7b-lora/adapter`.
- **3-epoch variant** (~65 min): `sbatch --export=ALL,RUN=qwen7b-lora-3ep,EPOCHS=3 train.sbatch`, then benchmark it with `RUN=lora-7b-3ep,LORA=/data/checkpoints/qwen7b-lora-3ep/adapter`.

- **Monitoring:** both losses should fall, and GPU utilisation should stay close to 100 %. Watch it in W&B (with `WANDB=1`) or in the Nebius console (worker VM → Metrics → GPU metrics). Example charts are in [02-training-and-inference.md](02-training-and-inference.md#training-results-7b).

**7. After: benchmark base + adapter** 🔐 (1 GPU, ~5 min)

```bash
cd ~/eval && sbatch --export=ALL,RUN=lora-7b,LORA=/data/checkpoints/qwen7b-lora/adapter eval.sbatch
```

**8. Compare** 🔐

```bash
cd ~/eval && /opt/venvs/vllm/bin/python compare.py /data/eval-runs/base-7b/metrics.json /data/eval-runs/lora-7b/metrics.json
```

Output from our run (725 test items):

| Metric | Before (base) | After (fine-tuned) | Change |
|---|---|---|---|
| Exact match: right tool + all arguments (headline) | 56.4 % | 68.2 % | +11.8 pts |
| Right tool chosen | 87.6 % | 91.1 % | +3.5 pts |
| Correctly answered without a tool | 60.3 % | 78.0 % | +17.7 pts |
| Malformed tool call (lower is better) | 0.4 % | 0.4 % | +0.0 pts |

## C. Inference: base vs fine-tuned

**1. Start one vLLM server for both models** 🔐 (holds 1 GPU until cancelled)

The server needs a free GPU, so it waits in the queue while a training job is running.

```bash
cd ~/eval && sbatch --export=ALL,LORA=/data/checkpoints/qwen7b-lora/adapter serve.sbatch
squeue                                       # note the node, e.g. worker-0
tail -f vllm-serve-<ID>.out                  # wait for "Application startup complete"
curl http://worker-0:8000/v1/models          # Qwen2.5-7B-Instruct (base) and toolace (fine-tuned)
```

**2. Compare side by side** 🔐

```bash
cd ~/eval && /opt/venvs/vllm/bin/python chat_compare.py --url http://worker-0:8000 --item toolace-00488#7 \
  --inputs /data/datasets/toolace-clean/eval_inputs.jsonl --expected /data/datasets/toolace-clean/eval_expected.jsonl
```

**3. Or send a plain OpenAI request** 🔐 Change only `"model"` to switch between `toolace` and `Qwen2.5-7B-Instruct`.

```bash
curl -s http://worker-0:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "toolace",
  "messages": [{"role": "user", "content": "What is the weather in Paris and in Tokyo?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
             "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
  "temperature": 0}' | python3 -m json.tool
```

- From your machine: `ssh -i ~/.ssh/id_ed25519 -L 8000:worker-0:8000 root@<login-node-ip>`, then use `http://localhost:8000`.
- Stop the server: `scancel <ID>`.

## D. Useful checks 🔐

| What | Command |
|---|---|
| Queue / job result | `squeue` · `sacct -j <ID> -o JobID,State,Elapsed,ExitCode` |
| Cancel a job | `scancel <ID>` |
| Metrics of one run | `cat /data/eval-runs/<RUN>/metrics.json` |
