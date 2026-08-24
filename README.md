# jetson-mate-llm-runbook

Runbook for deploying four Jetson Nano modules on a Seeed Jetson Mate carrier
(JetPack 4.6.6 / L4T 32.7.6) to serve the largest LLM possible for local
code-generation inference.

- Full procedure: [RUNBOOK.md](RUNBOOK.md)
- Primary deployment: pooled 4-GPU cluster running Qwen2.5-Coder-7B-Instruct Q4_K_M via the llama.cpp RPC backend (~1–3 tok/s)
- Fast-path alternative: one independent Qwen2.5-Coder-3B server per node (~2–3 tok/s each, 4 concurrent users)

## TL;DR pipeline

1. Flash JetPack 4.6.6 onto each module (Master slot, one at a time)
2. Headless tuning per node (no GUI, MAXN mode, 8 GB swap, static IPs)
3. Build llama.cpp b5050 once (CUDA sm_53 + RPC, gcc 8.5 toolchain) and replicate binaries
4. Start `rpc-server` on workers, `llama-server` on the head node

See RUNBOOK.md for prerequisites, exact commands, systemd units, benchmarks, and troubleshooting.
