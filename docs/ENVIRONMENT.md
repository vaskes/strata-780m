# Environment (llmhost2, 2026-10-20/21)

```
CPU      AMD Ryzen 7 8700G, 16 threads (8C/16T), Radeon 780M integrated (gfx1103, Phoenix)
RAM      58.2 GiB usable (61 070 256 kB MemTotal)
Kernel   7.0.0-38-generic
ROCm     host 7.2.4 (/opt/rocm-7.2.4), hipBLASLt 1.2.0  -- NOT usable for gfx1103
ROCm     engine build: rocm/dev-ubuntu-24.04:10.1.0-full, HIP 7.16.26385, hipBLASLt 1.4.1, clang 24
amdgpu   cwsr_enable=0
ttm      pages_limit = 13 369 344 pages (51.0 GiB GTT)
GPU      "AMD Radeon 780M Graphics", 6 CUs, 54.8 GB unified, Phoenix1 (0x7e66)
no NVIDIA driver present; nvidia-smi fails
```

llama.cpp build: `0.3.0-dev (build 10675, commit 90c26fcd4)`, GNU 13.3.0, x86-64.

Model on the host (already present, nothing downloaded for this work):

```
/opt/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf
  93.7 GB total, 59.5 GiB of experts
  sha256 matches setup.py UNSLOTH_IQ4_XS_SHARDS exactly (see docs/logs/sha.log)
/opt/models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf
```
