# 7. Soperator filesystem: the "jail"

Sources: Nebius blog "Soperator in open source explained" (https://nebius.com/blog/posts/soperator-in-open-source-explained), the Soperator 4.1.12 source, and our `installations/example/terraform.tfvars`.

## The problem it solves

In classic Slurm, every node has its **own** disk. Install `transformers` on node 1 and node 2 doesn't have it. Create a user on one node and you have to create it on all of them. Admins spend a lot of time keeping nodes identical (Ansible, images, NFS for `/home`).

## The idea in one sentence

**One shared filesystem (the "jail") is mounted on every login and worker node, and every SSH session and every Slurm job sees it as its root `/`.**

> "When users interact with the Slurm cluster (like browsing the filesystem in an SSH session or running a Slurm job), they actually see this shared filesystem as their root '/' directory." (blog)

So: `pip install` once on the login node → every worker has it. Write a checkpoint on worker-0 → worker-1 and the login node see it immediately.

## Analogy: a shared office with personal desks

- **The jail = the shared office floor.** Bookshelves, filing cabinets, software, everyone's projects. Anyone who walks in from any door sees the same room.
- **Each node = a different entrance door** into that same office (login, worker-0, worker-1).
- **Node-local parts = each person's own desk drawer.** `/dev` (the actual GPU in *this* machine), `/tmp`, `/proc`, `/run`. These can't be shared, because they describe *this* machine.
- **The building's machinery = the host / Kubernetes pod.** Slurm daemons, drivers, plumbing. Office workers can't touch it, even if they're "root" in the office.

## Shared vs node-local

| Shared (on the jail) | Node-local (per machine) |
|---|---|
| `/` itself: `/root`, `/opt`, `/usr`, `/etc`, `/home`*, `/data` | `/dev` (GPU devices, e.g. `/dev/nvidia0`) |
| Installed packages, venvs (`/opt/venvs/...`) | `/tmp` (in RAM) |
| Linux users and groups | `/proc`, `/run` |
| Job scripts, outputs, models, checkpoints | Low-level libraries, NVIDIA drivers, some system configs (bind-mounted from each host) |
| Slurm config: `/etc/slurm/*.conf` | Node-local submounts (e.g. `/mnt/local-data`) |

\* In our cluster `/home` is a separate NFS share mounted *inside* the jail (see "Our cluster" below).

> "While this approach won't work for sharing the rootfs of an arbitrary Linux system, it should cover what Slurm users typically need." (blog)

## How it works under the hood

The jail is a Kubernetes **PersistentVolume with `ReadWriteMany`** access (many pods can write at once). On Nebius it's a **Filestore** (shared network filesystem). Every login and worker pod mounts it at **`/mnt/jail`** (`consts.VolumeMountPathJail` in the source).

The pod's own root is a normal container (Slurm daemons, sshd). Users are then **moved into** `/mnt/jail` in two ways:

| Who | Mechanism | What it does |
|---|---|---|
| **SSH sessions** (login node) | OpenSSH `ChrootDirectory` | sshd puts the user's shell inside `/mnt/jail`, so `/mnt/jail` looks like `/` |
| **Slurm jobs** (workers) | SPANK plugin `chroot.so` | After slurmd forks the job but **before** your command runs, the plugin creates new Linux namespaces and calls `pivot_root` into `/mnt/jail`. Same trick container runtimes use. |

You can see the plugin in the generated `plugstack.conf`:
```
required chroot.so /mnt/jail
```
`required` = if the chroot fails, the job fails (it never runs outside the jail by accident).

**SPANK** = Slurm Plug-in Architecture for Node and job (K)control: hooks that run inside every job step.

```
                    Filestore (ReadWriteMany, 1 TiB)  ← the jail
                     │            │             │
        mounted at /mnt/jail in every pod
                     │            │             │
              ┌──────┴───┐  ┌─────┴────┐  ┌─────┴────┐
              │ login pod│  │worker-0  │  │worker-1  │
              │  sshd    │  │ slurmd   │  │ slurmd   │
              └────┬─────┘  └────┬─────┘  └────┬─────┘
       ChrootDirectory      chroot.so        chroot.so
                   │        (pivot_root)     (pivot_root)
                   ▼             ▼                ▼
              user sees /   job sees /       job sees /
              = jail        = jail           = jail
              + local /dev, /tmp, /proc, /run + local submounts
```

## Why it's safe: separation of concerns

> "Users can't accidentally break the Slurm cluster itself — all their actions are isolated within the jail. They can even have superuser rights inside the jail without having them on the host." (blog)

- Slurm daemons, sshd and the operator's config live **outside** the jail, in the pod's container.
- Users can `apt install`, break Python, delete files in `/opt`, and slurmd still runs.
- Being `root` in the jail ≠ root on the Kubernetes node.

## The trade-off: shared storage is slower

> "The performance of shared filesystems is significantly worse than that of local ones." (blog)

Fix: **submounts.** Mount faster local (or in-memory) storage at a path *inside* the jail. To the user it's just another folder, but it's not shared.

- **Jail submounts:** extra shared filesystems mounted inside the jail (e.g. a big `/mnt/data`). Useful above ~12 TiB.
- **Node-local jail submounts:** a per-node disk inside the jail. Fast, but **each node sees its own copy**.
- **In-memory submounts:** RAM-backed.

Under the hood, node-local pieces are layered from `/mnt/jail.upper` on each node (`consts.VolumeMountPathJailUpper`).

**Rule of thumb:**
- Anything every node must see (code, venvs, datasets, final checkpoints) → **jail**.
- Scratch, caches, temporary shards → **node-local submount** or `/tmp`.

## Our cluster (from `installations/example/terraform.tfvars`)

| Storage | Size | Where users see it | Shared? | Used for |
|---|---|---|---|---|
| `filestore_jail` | 1 TiB | `/` (incl. `/root`, `/opt`, `/data`) | ✅ all nodes | Venvs, models, datasets, checkpoints, scripts |
| `nfs_in_k8s` | 3.6 TiB (NETWORK_SSD_IO_M3) | `/home` | ✅ all nodes | Normal users' homes (empty in this PoC) |
| `node_local_jail_submounts` "local-data" | 1 TiB NETWORK_SSD per worker | `/mnt/local-data` | ❌ per node | Fast local scratch |
| `node_local_image_disk` | 930 GiB per worker | (container images) | ❌ per node | Docker / Enroot images |
| `filestore_accounting` | 512 GiB | (not visible to users) | – | Slurm accounting DB |
| Controller state | network SSD (`controller_state_on_filestore = false`) | (not visible) | – | slurmctld state / spool |
| `filestore_jail_submounts` | not used | – | – | (would be extra shared data mounts) |

**How this showed up in our PoC:**
- `python3 -m venv /opt/venvs/train` once on the login node → both workers could use it (shared jail).
- Models in `/data/models`, checkpoints in `/data/checkpoints` → both DDP ranks read and write the same paths.
- `wandb login` wrote `/root/.netrc` once → both nodes see the API key.
- `/etc/slurm/*.conf` lives in the jail → every slurmd reads the same config (see [03-slurm-gres.md](03-slurm-gres.md)).

Check it on the cluster:
```bash
df -h / /home /mnt/local-data       # jail, NFS, node-local
mount | grep -E ' / | /home | /mnt/local-data '
srun -N2 bash -c 'hostname; ls /opt/venvs'   # same content on both workers
srun -N2 bash -c 'hostname; touch /mnt/local-data/$(hostname); ls /mnt/local-data'   # different per node
```

## Limitations (from the blog, at the time it was written)

- Needs `ReadWriteMany` shared storage (NFS, Filestore, GlusterFS, OpenEBS).
- Sharing a root filesystem only works for the parts Slurm users normally need, not an arbitrary Linux system.
- Supported distros and versions were fixed (Ubuntu, specific Slurm / CUDA / kernel versions).
- Most operator containers ran **privileged** (chroot, namespaces, devices need it).

## Interview one-liner

> "Soperator mounts one shared ReadWriteMany volume, the jail, into every login and worker pod at /mnt/jail. sshd chroots SSH users into it and a SPANK plugin pivot_roots every Slurm job into it, so everyone sees the same root filesystem. You install once and it's everywhere. Device files, /tmp and /proc stay node-local, and performance-sensitive data goes on node-local submounts because shared storage is slower. Users can even be root in the jail without touching the Slurm daemons outside it."

**Memory hook:** *"One shared office, many doors, personal desk drawers. The building's machinery is off limits."*
