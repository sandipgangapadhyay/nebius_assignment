# Troubleshooting walkthrough: GPU workers crash-looping on topology (personal prep notes)

Purpose: understand, and be able to **re-verify step by step**, how the "missing topology label" problem was diagnosed and fixed, so it can be explained confidently in the demo.

> The assignment allows AI-assisted tools as long as you understand and can explain everything. If asked how you worked, it's fine to say you used an AI assistant as a tool. What matters is that you can walk through every step below and reproduce it.

---

## 1. One-paragraph summary (what to say)

After the cluster came up, both GPU worker pods crash-looped in their `worker-init` container. Its log said the node was **"not found in topology ConfigMap"** and the ConfigMap was **empty**. That ConfigMap is filled by a Soperator controller from **Kubernetes node labels**, which Nebius applies only to GPU nodes in an **InfiniBand GPU cluster**. The 1-GPU preset can't join one, so the labels never existed. **The log did not name the label.** I found the exact key, **`topology.nebius.com/tier-1`**, by tracing the ConfigMap name into the Soperator 4.1.12 source code: the controller that writes it and the constants it uses. I labelled both GPU nodes (both on one Ethernet "switch"), fixed a second issue (`tier-0` must not be set), then made the label permanent in Terraform.

---

## 2. The evidence chain: what each step told me

| # | What I looked at | What it showed | Conclusion |
|---|---|---|---|
| 1 | `kubectl -n soperator get pods` | `worker-0/1` stuck at `Init:5/6`, then `Init:CrashLoopBackOff`; controller, login, accounting fine | Problem is in the **worker start-up**, not the cluster |
| 2 | Which init container? `kubectl get pod worker-1 -o json` → `initContainerStatuses` | All passed except **`worker-init`** (exit code 1, restarting) | Focus on `worker-init` |
| 3 | **Earliest failure:** `kubectl -n soperator logs worker-1 -c worker-init --previous` | `Waiting for topology data for node …` · `Topology ConfigMap path: /tmp/slurm/topology-node-labels` · `not found after 180s` · **`Available nodes in ConfigMap: []`** | It waits 180 s for **topology data** that never arrives. The ConfigMap is **empty**. **No label name in the log** |
| 4 | `kubectl -n soperator get cm topology-node-labels -o yaml` | `data: null` | Confirms nothing writes into it |
| 5 | `kubectl get nodes -l slurm.nebius.ai/workload=gpu --show-labels` | GPU nodes have only `topology.kubernetes.io/region`, **no `topology.nebius.com/...`** | Nodes carry no network-position information |
| 6 | The init container's command and environment (pod spec) | `python3 /opt/bin/slurm/worker_init.py wait-controller wait-topology`, `TOPOLOGY_WAIT_TIMEOUT=180`, `SLURM_TOPOLOGY_PLUGIN=topology/tree` | The wait is a deliberate feature, tied to the **topology plugin** |
| 7 | Terraform recipe: `grep -rn topolog modules installations` | No option to turn the wait off; `main.tf` sets `topology/tree` for every non-GB300 cluster | Can't disable it from Terraform |
| 8 | **Soperator source, tag 4.1.12** (details in section 3) | Wait added whenever a topology plugin is set; the CRD defaults the plugin to `topology/tree`; the ConfigMap is filled **only from nodes with `topology.nebius.com/tier-0` or `tier-1` labels** | **Root cause:** those labels exist only on InfiniBand GPU-cluster nodes |
| 9 | Labelled both nodes, rechecked the ConfigMap | Filled within seconds with both nodes | Fix confirmed in principle |
| 10 | `worker-0` init log after the label | **New error:** `scontrol update … topology=default:root:eth-spine:eth-leaf` → `Requested topology configuration is not available` | A second, related bug (section 4) |
| 11 | Removed `tier-0`, kept `tier-1` only | `Topology applied successfully`; `sinfo` both `idle`; `scontrol show topology` → `SwitchName=eth-spine Nodes=worker-[0-1]` | Fixed |

**Method in one line:** start from the **earliest failure**, follow the one concrete identifier the log gives (the ConfigMap name), find **who writes it** and **what input they need**, then supply that input.

---

## 3. How the exact label key was found (the log doesn't say it)

The log only gave the **ConfigMap name** `topology-node-labels` (as a file path). From there, in the Soperator source at the deployed version:

```bash
git clone --depth 1 --branch 4.1.12 https://github.com/nebius/soperator.git && cd soperator
```

**a) Who writes the ConfigMap?**

```bash
grep -rn 'topology-node-labels' --include='*.go' .
# internal/consts/configmap.go:34:   ConfigMapNameTopologyNodeLabels = "topology-node-labels"
# internal/controller/topologyconfcontroller/nodetopology_controller.go   ← the controller that writes it
```

**b) Which nodes does that controller use?** `internal/controller/topologyconfcontroller/nodetopology_controller.go`, lines 173–181:

```go
// shouldProcessNode checks if the node has the required tier-1 label
func (r *NodeTopologyReconciler) shouldProcessNode(node *corev1.Node, …) bool {
    _, hasTierZero := node.Labels[r.tierZeroLabel()]
    _, hasTierOne  := node.Labels[r.tierOneLabel()]
    if !hasTierZero && !hasTierOne { … "Node missing one of tier-0 or tier-1 label, skipping" … return false }
```

with `tierOneLabel()` = `topologyLabelPrefix + consts.TierOneSuffix`.

**c) What is the prefix / suffix?**

```bash
grep -rn 'DefaultTopologyLabelPrefix *=\|TierOneSuffix\|TierZeroSuffix' internal/consts
# internal/consts/label.go:38:  DefaultTopologyLabelPrefix = "topology.nebius.com"
# internal/consts/label.go:39:  TierZeroSuffix             = "/tier-0"
# internal/consts/label.go:40:  TierOneSuffix              = "/tier-1"
sed -n 146,149p cmd/main.go
#   topologyLabelPrefix := os.Getenv("TOPOLOGY_LABEL_PREFIX")
#   if topologyLabelPrefix == "" { topologyLabelPrefix = consts.DefaultTopologyLabelPrefix }
```

→ the key is **`topology.nebius.com/tier-1`**, unless the operator sets `TOPOLOGY_LABEL_PREFIX`. Ours doesn't: the ConfigMap filled as soon as nodes had that key. The unit tests use the same keys (`nodetopology_controller_test.go`: `topology.nebius.com/tier-1: "leaf00"`), and on InfiniBand clusters Nebius sets the value to the real switch ID.

**d) Why the wait happens at all.** `internal/controller/nodesetcontroller/reconcile.go:398`: `topologyPluginEnabled := topologyPlugin != ""` → `internal/render/worker/container.go:36` appends `"wait-topology"`. The CRD default (`api/v1/slurmcluster_types.go:179`) is `topology/tree`, so it can't be left empty.

**The value** (`eth-spine`) is free-form: just a switch name. Both nodes get the **same** value, so Slurm sees them under one switch, which is true for 2 VMs on one Ethernet network.

---

## 4. The second issue: why `tier-1` only (no `tier-0`)

First attempt: `tier-0=eth-leaf` + `tier-1=eth-spine`. `worker-1` succeeded (it registered before the topology was rebuilt); `worker-0` then failed:

```
INFO  Running: scontrol update nodename=worker-0 … topology=default:root:eth-spine:eth-leaf …
ERROR scontrol update failed (rc=1): slurm_update error: Requested topology configuration is not available
```

| Check | Showed |
|---|---|
| `kubectl -n soperator get cm topology-config` | Slurm's `topology.conf` had switches **`root`** and **`eth-spine`** only, **no `eth-leaf`** |
| `internal/controller/topologyconfcontroller/topology_graph.go:179` | *"In case of tier-0 label provided – we ignore it … (tier-0 used for defining block, not IB topology)"* |
| `images/worker/worker_init.py` → `_format_tier_topology` | Builds the switch path from **all** tiers, **including tier-0** → `root:eth-spine:eth-leaf` |
| `nodetopology_controller.go:178` | Needs **either** tier-0 **or** tier-1 |

→ The two parts of Soperator disagree about `tier-0`. With **`tier-1` only**, both produce `root:eth-spine`. (Worth suggesting upstream.)

---

## 5. The fix

**Live (to unblock):**

```bash
kubectl label nodes -l slurm.nebius.ai/workload=gpu topology.nebius.com/tier-1=eth-spine
```

**Permanent (Terraform), so replaced nodes get it too:**
- `modules/k8s/locals.tf`: `node_group_ethernet_topology_label_v2` = `{ "topology.nebius.com/tier-1" = "eth-spine" }`, **only for GPU workers with `gpu_cluster == null`** (InfiniBand clusters are untouched; Nebius labels those).
- `modules/k8s/k8s_ng_workers_v2.tf`: merged into the worker node group's `template.metadata.labels`.
- `terraform apply`: updated in place, no node recreation; the next `terraform plan` shows 0 changes.

---

## 6. Re-verify it yourself now

```bash
# Kubernetes (Mac)
kubectl config use-context nebius-startup-slurm
kubectl get nodes -l slurm.nebius.ai/workload=gpu -L topology.nebius.com/tier-1      # eth-spine on both
kubectl -n soperator get cm topology-node-labels -o jsonpath='{.data}'               # both nodes → {"tier-1":"eth-spine"}
kubectl -n soperator get cm topology-config -o jsonpath='{.data.topology\.conf}'     # SwitchName=eth-spine Nodes=worker-0,worker-1
kubectl -n soperator logs worker-0 -c worker-init | grep -E 'Found topology|Topology applied'
kubectl -n soperator-system get deploy -o yaml | grep -A1 TOPOLOGY_LABEL_PREFIX       # nothing → default prefix in use

# Slurm (login node)
scontrol show topology        # SwitchName=eth-spine Level=0 Nodes=worker-[0-1] / SwitchName=root …
sinfo -N                       # both idle

# Terraform (Mac, installations/example)
grep -n 'topology.nebius.com' ../../modules/k8s/locals.tf
terraform plan                 # no change to the worker node groups

# Source (anywhere)
git clone --depth 1 --branch 4.1.12 https://github.com/nebius/soperator.git
grep -rn 'DefaultTopologyLabelPrefix *=\|TierOneSuffix' soperator/internal/consts
sed -n 173,181p soperator/internal/controller/topologyconfcontroller/nodetopology_controller.go
sed -n 175,182p soperator/internal/controller/topologyconfcontroller/topology_graph.go
```

The original failure log isn't on the cluster any more (the pods were restarted). The excerpts above come from the session, and the same text is in `FINDINGS.md` #12.

---

## 7. Likely questions and short answers

| Question | Answer |
|---|---|
| How did you know it was a topology problem? | The earliest failing container, `worker-init`, said it was waiting for topology data and timed out; the ConfigMap it reads was empty |
| Did the log tell you the label name? | No. It gave the ConfigMap name. I traced that name in the Soperator source to the controller that writes it, which reads `topology.nebius.com/tier-0` / `tier-1` node labels (prefix and suffix are constants in `internal/consts/label.go`) |
| Why were the labels missing? | Nebius sets them for GPU nodes in an InfiniBand GPU cluster; the 1-GPU preset can't join one |
| Why not just disable the wait? | It runs whenever a topology plugin is set, and the CRD defaults the plugin to `topology/tree`. The recipe has no switch for it, and editing the init container would be overwritten by the operator |
| Why not just raise the timeout? | The data never arrives without labels, so it would only delay the failure |
| What does `eth-spine` mean? | A free-form switch name. Both nodes share it because they're on one Ethernet network; on IB clusters the value is the real switch ID |
| Why no `tier-0`? | Soperator's topology builder ignores `tier-0` for tree topology, but the worker script includes it in the switch path, so Slurm rejects the path. `tier-1` alone keeps both consistent |
| How did you make it permanent? | A Terraform label on the worker node-group template, only for GPU workers without a GPU cluster, applied in place |
| What would you suggest upstream? | Skip `wait-topology` (or fall back to a default switch) when no node has tier labels, and make `worker_init.py` ignore `tier-0` for `topology/tree` |
