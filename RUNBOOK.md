# Runbook — 4x Jetson Nano on Seeed Jetson Mate, JetPack 4.6.6, Local LLM Coding Assistant

Audience: operator comfortable with Linux CLI. Total effort: ~1 working day (flashing 4x ~30 min, OS tuning, one 3 h toolchain build, one ~85 min llama.cpp build, model download).

---

## 1. Goal and architecture

Deploy the **largest code-generation LLM possible** for fully local inference on a Seeed Jetson Mate carrying four Jetson Nano (4 GB) modules, all running **JetPack 4.6.6 (L4T 32.7.6, Ubuntu 18.04, CUDA 10.2)**.

### Hardware reality check (set expectations)

| Constraint | Value | Consequence |
|---|---|---|
| RAM per node | 4 GB LPDDR4, shared GPU/CPU | Max ~2.5 GB usable per node for model tensors |
| GPU | 128-core Maxwell, sm_53, CUDA 10.2 | No bf16, no modern kernels; needs patched llama.cpp |
| Memory bandwidth | 25.6 GB/s | Single node tops out around 6–7 tok/s on a 1B model |
| Interconnect | Onboard KSZ9896C Gigabit switch | Pooling 4 nodes works, but network hops cost latency |

Measured reference (TinyLlama-1.1B Q4_K_M, GPU offloaded, single Nano): **pp512 ≈ 80 t/s, tg128 ≈ 6.7 t/s**.

### Deployment modes

| Mode | Topology | Largest model | Expected generation speed | Use when |
|---|---|---|---|---|
| **A — Pooled cluster** (primary goal) | One 7B model sharded across all 4 GPUs via llama.cpp RPC backend | **Qwen2.5-Coder-7B-Instruct Q4_K_M (~4.7 GB)**; stretch: 14B Q4_K_M (~9 GB) | ~1–3 tok/s | You want the biggest/best model; latency tolerance |
| **B — Per-node serving** (fast path / fallback) | One independent server per node | Qwen2.5-Coder-3B-Instruct Q4_K_M (~1.9 GB) per node | ~2–3 tok/s per node, 4 concurrent sessions | You want responsiveness or multiple users |

> Important: pooling 4 nodes gives you a **bigger** model, **not faster** tokens. RPC splits layers across machines; every token traverses the network. The cluster runs at the speed of its weakest hop.

Model choices are constrained to architectures supported by llama.cpp b5050 (April 2025) and by CUDA 10.2: Qwen2.5-Coder, DeepSeek-Coder, CodeLlama, StarCoder2 all qualify. Qwen2.5-Coder gives the best quality-per-parameter for coding today.

---

## 2. Bill of materials and prerequisites

- Seeed Jetson Mate carrier + case/fan, **65 W (or higher) USB-PD adapter (20 V/3 A+)**
- **4x Jetson Nano production modules** — WARNING: Jetson Nano Developer Kit **A02** modules do NOT work on Jetson Mate. Use A03/A06-era modules (production module with 16 GB eMMC, or devkit modules with SD cards)
- Host PC for flashing: **native Ubuntu 18.04 or 16.04 x86_64** (required by SDK Manager for JetPack 4.x). A VM must be VMware Workstation Player (VirtualBox fails to flash)
- Micro-USB cable, Ethernet cable to a router/switch (DHCP with internet access)
- Optional but recommended: USB 3.0 SSD per node (or one shared) for swap/models; fast A1-class microSD cards if your modules are devkit-type
- NVIDIA developer account (SDK Manager download)

Network plan used throughout this document (adjust to your LAN):

| Node | Hostname | IP (DHCP reservation recommended) | Role |
|---|---|---|---|
| Slot 0 "Master" | `jm-n0` | 192.168.1.240 | RPC client + llama-server + model storage |
| Slot 1 | `jm-n1` | 192.168.1.241 | rpc-server worker |
| Slot 2 | `jm-n2` | 192.168.1.242 | rpc-server worker |
| Slot 3 | `jm-n3` | 192.168.1.243 | rpc-server worker |

---

## 3. Phase 0 — Hardware assembly

1. Seat each Nano module firmly into its SO-DIMM slot (Master = leftmost slot). Screw down standoffs.
2. Connect the case fan, close the enclosure.
3. Do NOT insert all modules for first boot; flashing is done one module at a time in the **Master slot**.

Checklist: fan connected, PSU is USB-PD 20 V capable, Ethernet plugged into the Mate's RJ45.

---

## 4. Phase 1 — Flash JetPack 4.6.6 onto each module

All modules can **only** be flashed while installed in the Master slot. Flash and configure them **one at a time**, then move to a worker slot.

On the Ubuntu 18.04 host:

1. Install SDK Manager: <https://developer.nvidia.com/sdk-manager> and log in.
2. Target hardware: **Jetson Nano**; JetPack version: **4.6.6** (latest 4.6.x offering).
3. Keep both **Jetson OS** and **JetPack SDK Components** checked (components install CUDA 10.2 on the module).
4. Place the first module in the Master slot. Force USB recovery: short the two FC_REC/GND pins (jumper, see Seeed wiki photo), connect the micro-USB cable to the host, press the **wake button** to power on, then remove the jumper.
5. Verify the host sees it: `lsusb | grep -i nv` → "NVIDIA Corp. APX".
6. In SDK Manager: Manual Setup, **flash**, wait for OS flash to finish.
7. Open serial console on the host for first-boot configuration:
   ```
   sudo apt install minicom
   dmesg | grep tty          # device is typically /dev/ttyACM0
   sudo minicom -b 9600 -D /dev/ttyACM0
   ```
8. Complete the initial Ubuntu setup in minicom (create user `jetson`, hostname `jm-n0`, enable SSH when prompted).
9. Back in SDK Manager, enter the credentials and let it **install SDK components** (CUDA 10.2, cuDNN, TensorRT).
10. Power down, move this module to its worker slot, place the next blank module in Master, repeat steps 4–9 for `jm-n1` … `jm-n3` (set matching hostnames during OOBE).

> Alternative without SDK Manager: flash manually from any Linux x86 host using the L4T R32.7.6 driver package + sample rootfs (`Linux_for_Tegra/flash.sh jetson-nano-emmc mmcblk0p1`), then install CUDA on-device with `sudo apt update && sudo apt install cuda-toolkit-10-2`.

**Verify on each node (SSH or serial):**

```
cat /etc/nv_tegra_release          # expect: R32 (release), REVISION: 7.6  → JetPack 4.6.6
nvcc --version                     # expect: release 10.2, V10.2.300
```

If `/etc/nv_tegra_release` shows < 7.6 (older 4.6.x image), upgrade in place:

```
sudo apt update && sudo apt full-upgrade -y && sudo reboot
```

If `nvcc` is not found, fix PATH:

```
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. Phase 2 — Per-node OS tuning (repeat on ALL four nodes)

Goal: headless, max performance, enough virtual memory for model loads.

```bash
# Headless: kill the desktop, frees ~600-800 MB RAM (REQUIRED on 4 GB)
sudo systemctl set-default multi-user.target

# Performance: MAXN power mode (10 W) and pinned clocks
sudo nvpmodel -m 0                 # verify: sudo nvpmodel -q  → MODE_10W/MAXN
sudo jetson_clocks

# Persist clocks after reboot
sudo jetson_clocks --store

# Extra swap (in addition to default zram). Put it on a USB3 SSD if you have one,
# otherwise eMMC/SD is acceptable for lab use.
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-llm.conf

# Static IP via NetworkManager (example for jm-n1; adjust address per node)
sudo nmcli con mod "Wired connection 1" ipv4.method manual \
  ipv4.addresses 192.168.1.241/24 ipv4.gateway 192.168.1.1 \
  ipv4.dns "192.168.1.1 1.1.1.1"
sudo systemctl restart NetworkManager
```

Reboot, then verify from any machine: `ssh jetson@192.168.1.241`, and on-node `free -h` shows ~3.4 GB available and swap active.

Install shared basics on every node:

```bash
sudo apt update
sudo apt install -y git curl wget jq ethtool libcurl4-openssl-dev python3-pip
pip3 install jetson-stats        # provides `jtop` (may take a while on py3.6)
```

Sanity-check the GPU on each node:

```bash
sudo tegrastats                  # or: jtop   → should show GPU, 3956 MiB RAM
```

---

## 6. Phase 3 — Build llama.cpp (CUDA sm_53 + RPC backend)

Recent llama.cpp needs patches to compile against nvcc 10.2 (bf16 intrinsics don't exist there). We pin the proven combination: **commit `23106f9` (release b5050)** + **gcc 8.5** following the community-proven procedure at <https://github.com/kreier/llama.cpp-jetson>.

Strategy: do the full build **once on jm-n0**, then copy binaries to the other three nodes (identical OS/arch).

### 6.1 Toolchain on jm-n0 (~3.5 h, unattended)

The apt `gcc-8` (8.4) fails on `ggml-quants.c: vld1q_s8_x4`; gcc 8.5 from source is required. Install it to a replicable prefix:

```bash
sudo apt install -y build-essential software-properties-common \
                    libgmp-dev libmpfr-dev libmpc-dev
cd /tmp
wget http://ftp.gnu.org/gnu/gcc/gcc-8.5.0/gcc-8.5.0.tar.gz
tar xf gcc-8.5.0.tar.gz && cd gcc-8.5.0
./contrib/download_prerequisites
mkdir build && cd build
../configure --prefix=/opt/gcc-8.5 --enable-languages=c,c++ --disable-multilib
make -j$(nproc)                   # ~3 h
sudo make install
```

CMake >= 3.14 (Ubuntu 18.04 ships 3.10). Fastest reliable way is the Kitware repo:

```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
  gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | \
  sudo tee /etc/apt/sources.list.d/kitware.list
sudo apt update && sudo apt install -y cmake
cmake --version                   # >= 3.24 preferred
```

(Fallback: `pip3 install cmake`.)

### 6.2 Build llama.cpp (~85 min)

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp llama.cpp && cd llama.cpp
git checkout 23106f9              # b5050, known-good with CUDA 10.2
git checkout -b nano-cuda
```

Apply the six file edits documented (with exact diffs) in the **Procedure** section of <https://github.com/kreier/llama.cpp-jetson>. Summary of what they do — do not skip any:

- `CMakeLists.txt` — force `CMAKE_CUDA_ARCHITECTURES=53`, relax compiler checks
- `ggml/CMakeLists.txt` — CUDA 10.2-compatible flags
- `ggml/src/ggml-cuda/common.cuh`, `fattn-common.cuh`, `fattn-vec-f32.cuh`, `fattn-vec-f16.cuh` — guard/remove `__nv_bfloat16` usage (unsupported by nvcc 10.2), fall back to fp16/fp32 paths

Configure and build with **both** backends we need (CUDA for compute, RPC for clustering):

```bash
export CC=/opt/gcc-8.5/bin/gcc
export CXX=/opt/gcc-8.5/bin/g++
export PATH=/opt/gcc-8.5/bin:$PATH

cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_RPC=ON \
  -DLLAMA_CURL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=53
cmake --build build --config Release -j4
```

Smoke test on jm-n0 (downloads a small GGUF via the built-in downloader):

```bash
./build/bin/llama-cli -hf TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:Q4_K_M -ngl 99 -n 48 \
  -p "Write a C function that reverses a string."
```

Success criteria: log shows `found 1 CUDA devices: Device 0: NVIDIA Tegra X1, compute capability 5.3` and coherent text output at roughly 5–7 tok/s. Confirm the RPC binary exists: `ls build/bin/rpc-server`.

### 6.3 Replicate to jm-n1..n3

```bash
# on jm-n0
cd ~/llama.cpp/build/bin
tar czf /tmp/llama-nano-b5050.tar.gz .

for h in 241 242 243; do scp /tmp/llama-nano-b5050.tar.gz jetson@192.168.1.$h:/tmp/; done
```

On **each** worker node:

```bash
sudo mkdir -p /opt/llama/bin
sudo tar xzf /tmp/llama-nano-b5050.tar.gz -C /opt/llama/bin
sudo chmod +x /opt/llama/bin/*
printf '%s\n' 'export PATH=/opt/llama/bin:$PATH' \
              'export LD_LIBRARY_PATH=/opt/llama/bin:$LD_LIBRARY_PATH' | \
  sudo tee /etc/profile.d/llama.sh
source /etc/profile.d/llama.sh
rpc-server --help                 # must print usage without library errors
```

> Version skew rule: the RPC client and ALL rpc-server processes must come from the **exact same build/commit**. If you ever rebuild one node, redeploy to all four. Mismatch manifests as a silent hang during model load.

---

## 7. Phase 4 — Mode A: pooled 4-GPU cluster (primary deployment)

Topology: `rpc-server` on jm-n1/n2/n3 (and optionally jm-n0 too); `llama-server` client on jm-n0 holding the model file. The client sees 4 CUDA devices (its own + 3 remote) and auto-splits weights and KV cache proportionally to free memory.

### 7.1 Download the model on jm-n0

```bash
mkdir -p ~/models && cd ~/models
llama-cli --list-repo-files Qwen/Qwen2.5-Coder-7B-Instruct-GGUF   # check exact names
# Preferred: let the built-in downloader fetch the Q4_K_M file
llama-cli -hf Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M --download-only || true
# Or direct (verify filename on the HF page; large quants may be split into parts):
wget -c https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

Disk note: 16 GB eMMC fits the 4.7 GB model + caches, but a USB3 SSD is more comfortable. If you use a drive, mount it at `/mnt/ssd` and point model paths and RPC caches there.

### 7.2 Start workers (jm-n1, jm-n2, jm-n3)

Manual test first, on each worker:

```bash
rpc-server -H 0.0.0.0 -p 50052 -m 2600 -c
```

Flags: `-m 2600` caps the worker's device buffer at 2600 MB (protects the OS on a 4 GB board — tune 2200–2800 after watching `free -h`), `-c` enables a local weight cache in `~/.cache/llama.cpp/rpc` (skip `-c` if eMMC is tight; it costs ~1.2 GB per worker but makes reloads much faster).

Then make it durable via systemd, on each worker:

```ini
# /etc/systemd/system/rpc-server.service
[Unit]
Description=llama.cpp RPC worker
After=network-online.target
Wants=network-online.target

[Service]
User=jetson
Environment=LD_LIBRARY_PATH=/opt/llama/bin
ExecStart=/opt/llama/bin/rpc-server -H 0.0.0.0 -p 50052 -m 2600 -c
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now rpc-server
systemctl status rpc-server       # expect: "Starting RPC server", endpoint :50052, CUDA0 Tegra X1
```

### 7.3 Launch the pooled model on jm-n0

Test run (foreground):

```bash
llama-server \
  -m ~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 \
  -c 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --threads 4 \
  --rpc 192.168.1.241:50052,192.168.1.242:50052,192.168.1.243:50052
```

Notes:
- jm-n0's own GPU joins the pool automatically; default splitting is proportional to free memory. Override with `--tensor-split a,b,c,d` if one node OOMs (order = local device first, then RPC endpoints in listed order).
- `-ngl 99` puts every layer in the pool; `-c 4096` keeps KV cache modest; q8_0 KV halves its footprint (Maxwell-safe, supported in b5050).

First load takes minutes (≈4.7 GB streamed over GbE + CUDA JIT). Success looks like, in the client log:

```
load_tensors:  RPC[192.168.1.241:50052] model buffer size = ~1200 MiB
load_tensors:  RPC[192.168.1.242:50052] model buffer size = ~1200 MiB
load_tensors:  RPC[192.168.1.243:50052] model buffer size = ~1200 MiB
load_tensors: CUDA0 model buffer size = ~1200 MiB
```

And on each worker, `jtop` shows GPU memory in use. If a worker shows nothing, its shard was never assigned — see Troubleshooting.

Make it durable on jm-n0:

```ini
# /etc/systemd/system/llama-server.service
[Unit]
Description=llama.cpp server (pooled cluster client)
After=network-online.target rpc-server.service
Wants=network-online.target

[Service]
User=jetson
Environment=LD_LIBRARY_PATH=/opt/llama/bin
ExecStartPre=/bin/sh -c 'until nc -z 192.168.1.241 50052 && nc -z 192.168.1.242 50052 && nc -z 192.168.1.243 50052; do sleep 2; done'
ExecStart=/opt/llama/bin/llama-server -m /home/jetson/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 4096 --cache-type-k q8_0 --cache-type-v q8_0 --threads 4 --rpc 192.168.1.241:50052,192.168.1.242:50052,192.168.1.243:50052
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now llama-server
```

### 7.4 Stretch goal: 14B

With all four workers capped at `-m 2800` and jm-n0 contributing ~2.5 GB, the pool holds ~11 GB — enough for **Qwen2.5-Coder-14B-Instruct Q4_K_M (~9.0 GB)** plus a small KV cache. Expect sub-1 tok/s generation and multi-minute prompt processing; treat as an experiment proving the ceiling, not a daily driver. Reduce context to `-c 2048` and consider IQ4_XS (~8.0 GB) to relieve pressure.

---

## 8. Phase 5 — Mode B: one model per node (fast path / fallback)

Skip the entire Phase 3 source build: the prebuilt CUDA binaries (no `rpc-server`) install in one minute per node:

```bash
curl -fsSL https://kreier.github.io/llama.cpp-jetson.nano/install.sh | bash && source ~/.bashrc
```

Per node (each gets a different model/port if you like, or keep them identical behind a load balancer):

```bash
llama-server -m ~/models/qwen2.5-coder-3b.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 --threads 4
# fetch once per model type:
# llama-cli -hf Qwen/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M --download-only
```

Four nodes → four independent OpenAI-compatible endpoints (`http://192.168.1.24x:8080/v1`). Put nginx round-robin in front if you need one URL. This mode yields ~2–3 tok/s per session with up to 4 simultaneous users — usually the more pleasant experience despite the smaller model.

---

## 9. Validation and benchmarks

Functional test (from any machine, against Mode A on jm-n0):

```bash
curl http://192.168.1.240:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a bash one-liner to find the 10 largest files under /var."}],"max_tokens":250,"temperature":0.2}'
```

Acceptance criteria: valid JSON stream, coherent code, no worker OOM in `journalctl -u rpc-server` (all three workers), client log shows buffers on 4 devices.

Throughput measurement:

```bash
# Single-node baseline (Mode B)
llama-bench -m ~/models/qwen2.5-coder-3b.gguf -ngl 99 -p 512 -n 128
# Pooled cluster (Mode A) — llama-bench accepts --rpc exactly like llama-server
llama-bench -m ~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf -ngl 99 -p 512 -n 128 \
  --rpc 192.168.1.241:50052,192.168.1.242:50052,192.168.1.243:50052
```

Expected ballpark (accept ±50%):

| Configuration | pp512 (prompt) | tg128 (generation) |
|---|---|---|
| 1B Q4_K_M, single node | ~80 t/s | ~6.7 t/s (measured reference) |
| 3B Q4_K_M, single node | ~25–35 t/s | ~2–3 t/s |
| 7B Q4_K_M, pooled 4x Nano | ~15–35 t/s | ~1–3 t/s |
| 14B Q4_K_M, pooled 4x Nano | <15 t/s | <1 t/s |

Thermal check under sustained load: `jtop` — GPU temp should stay < 85 °C with the case fan spinning; watch for throttling in `sudo tegrastats` output (clock drops below the jetson_clocks pins).

---

## 10. Operations

- Logs: `journalctl -u llama-server -u rpc-server -f` (centralize with `@jm-n0` aliases in `~/.ssh/config`)
- Health probe: `curl -fs http://192.168.1.240:8080/health` from cron; alert on failure
- Rolling restart of a worker: `sudo systemctl stop rpc-server` → the client will error on next request; restart `llama-server` after the worker returns (RPC has no live rejoin)
- Changing models (Mode A): edit the `-m` path in `/etc/systemd/system/llama-server.service`, `sudo systemctl daemon-reload && sudo systemctl restart llama-server`
- After any reboot, systemd brings workers up first; the `ExecStartPre` gate on jm-n0 prevents the client from starting early
- Security: the RPC protocol has **no authentication**. Keep the cluster on a trusted LAN, never port-forward 50052. The Mate's internal switch plus your router firewall is the boundary
- Updating llama.cpp: re-check <https://github.com/kreier/llama.cpp-jetson> for a newer proven commit; rebuild on jm-n0 and redeploy to all nodes together (version skew rule)

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Module never appears in `lsusb` during flash | Not in recovery mode, or A02 module | Redo FC_REC jumper; A02 modules are incompatible with Jetson Mate |
| SDK Manager won't run | Host not Ubuntu 16/18.04 | Use native 18.04 or VMware Player; or use manual `flash.sh` method |
| `minicom` shows nothing | Wrong port/baud | `dmesg \| grep tty`; use `-b 9600 -D /dev/ttyACM0` |
| `nvcc: command not found` | PATH | Section 4 snippet |
| Compile error `vld1q_s8_x4` in `ggml-quants.c` | Used apt gcc-8 (8.4) | Build gcc 8.5 per §6.1 |
| Compile errors mentioning `__nv_bfloat16` / `fattn-*` | Patch edits skipped/wrong commit | Re-checkout `23106f9`, redo all six edits |
| Client hangs forever at model load | Client/server build mismatch, or worker down | Verify all nodes report same build; `nc -z` each worker:50052 |
| Worker asserts `ne[0] % 512 == 0` | Quant/model shape not RPC-friendly | Switch quant (Q8_0) or model family (Qwen/Llama shapes are safe) |
| Worker OOM / killed during load | `-m` cap too high, or GUI still running | Lower `-m` to 2200; confirm multi-user.target; check `free -h` |
| One worker idle (no GPU mem) | Shard went elsewhere (client had room) or unreachable endpoint | Check client log device list; force split with `--tensor-split` |
| Very slow (<0.5 t/s) | Wi-Fi/link at 100 Mbps, or thermal throttling | `ethtool eth0` must show Speed: 1000Mb/s on all nodes; check temps |
| First load extremely slow every time | RPC weight cache disabled/full disk | Enable `-c` on workers; free eMMC (`sudo apt clean`, prune journals) |
| Random reboots under load | Undersized PSU | Must be genuine 65 W+ USB-PD, 20 V profile |

---

## 12. References

- Seeed Jetson Mate getting started (flash order, recovery jumper, minicom): <https://wiki.seeedstudio.com/Jetson-Mate/>
- JetPack 4.6.6 / L4T 32.7.6 archive: <https://developer.nvidia.com/jetpack-sdk-466> and <https://developer.nvidia.com/embedded/jetpack-archive>
- llama.cpp on Jetson Nano with CUDA 10.2 (patch procedure, benchmarks): <https://github.com/kreier/llama.cpp-jetson>
- Prebuilt CUDA binaries for Nano (Mode B): <https://github.com/kreier/llama.cpp-jetson.nano>
- llama.cpp RPC backend docs: <https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md>
- Models: <https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF>, <https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF>

---

## Appendix A — Command cheat sheet

```bash
# Status
cat /etc/nv_tegra_release && nvcc --version        # JP 4.6.6 / CUDA 10.2 on each node
sudo nvpmodel -q                                   # must be MAXN
jtop                                               # thermals, RAM, GPU

# Workers (n1..n3)
sudo systemctl restart rpc-server && journalctl -u rpc-server -f

# Cluster (n0)
sudo systemctl restart llama-server && journalctl -u llama-server -f
curl -s http://127.0.0.1:8080/v1/models

# Quick interactive session, pooled
llama-cli -m ~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf -ngl 99 -c 4096 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --rpc 192.168.1.241:50052,192.168.1.242:50052,192.168.1.243:50052
```
