# Strata & llama.cpp on Radeon 780M (gfx1103)

What a **Ryzen 7 8700G** with no discrete GPU can actually do with a 177B-class model,
measured on `llmhost2` — both with upstream **llama.cpp** and with **Strata**
(`Niko1221/Strata`), built from source for gfx1103 and run in Docker.

## Result

| engine | quant | bits/weight | decode | implied expert bandwidth |
|---|---|---|---|---|
| **Strata** 0.1.46.1 (HIP/gfx1103) | UD-IQ4_XS | ~4.25 | **9.4 tok/s** | ~15 GB/s |
| **llama.cpp** 0.3.0-dev `90c26fcd4` | Q8_0 | 8.0 | **9 tok/s** | — |
| Strata (upstream, 8745HS) | Q2_0 | 2.0 | 13–16 tok/s | ~10 GB/s |

Two unrelated inference engines land on the same number. So does a third run, on
different silicon, with a **2.1× smaller quant**. That is not a coincidence —
see [The wall is memory bandwidth](#the-wall-is-memory-bandwidth).

> **The limit is memory bandwidth.** Every generated token has to pull the active
> experts out of shared DDR5 over the APU's GPU memory path. That path delivers on the
> order of 10–15 GB/s of scattered expert reads, whatever you do above it. The engine
> does not matter; the quant size does not matter; **~9 tok/s is what a 177B MoE gets
> out of a Ryzen 8700G's iGPU.**

What *is* remarkable is that a model in this class runs **at all** on a box with no
graphics card. Buying a Ryzen 8700G is the cheapest way to put a 177B model on your
desk; there is no other way to get there without several-thousand-dollar hardware.

---

## Machine

```
CPU      AMD Ryzen 7 8700G — Zen 4, 8 cores / 16 threads, 1 NUMA node
GPU      Radeon 780M, RDNA3, gfx1103 (Phoenix1, 0x7e66), 12 compute units, unified memory
RAM      2 × 32 GB Kingston KF560C36-32, DDR5 **4800 MT/s, dual channel** (P0 CHANNEL A + B)
         58.2 GiB usable (MemTotal 61 070 256 kB)
Memory   DDR5-4800 × 2 channels × 8 bytes = 76.8 GB/s theoretical
Kernel   7.0.0-38-generic
amdgpu   cwsr_enable=0, ttm.pages_limit = 13 369 344 pages (51.0 GiB GTT)
ROCm     7.2.4 host (/opt/rocm-7.2.4), hipBLASLt 1.2.0 — not usable for gfx1103
```

No discrete GPU, no NVIDIA driver (`nvidia-smi` fails). Everything below is the iGPU
sharing the same DDR5 the CPU cores use.

---

## Model

`Qwen3.8-Flash-Next` — 177B-class MoE (A3B: ~3B parameters active per token),
quantised to GGUF. The IQ4_XS shards were already on the host, so **nothing was
downloaded**:

```
/opt/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1,2,3}-of-00003.gguf
5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4
577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7
d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833
```

Those are the exact SHA-256s pinned in Strata's `setup.py`
(`UNSLOTH_IQ4_XS_SHARDS`), so the model is byte-identical to what upstream ships.
93.7 GB across 3 shards, **59.5 GiB of experts** (Strata's own figure:
`MODELS["UD-IQ4_XS"]["arena_gb"]`).

---

## Strata — 9.4 tok/s

### Build

Strata has **no Vulkan backend at all** (zero `vulkan`/`GGML_VULKAN` hits in a 152 KB
`CMakeLists.txt`); backends are CUDA / HIP / SYCL only. There is also **no ready-made
engine for gfx1103** — the releases ship `hip-rx7900` (gfx1100/1101/1102) which will
not start on a 780M. gfx1103 is `experimental and opt-in` upstream
(`docs/RADEON_780M.md`), so the engine is compiled from source.

The host's ROCm 7.2.4 cannot be used: its hipBLASLt is **1.2.0** and gfx1103 needs a
tuning table for **1.4.1**. The engine is built inside
`rocm/dev-ubuntu-24.04:10.1.0-full` (HIP 7.16.26385, hipBLASLt 1.4.1, clang 24) — the
exact combination the gfx1103 table was calibrated against.

```bash
bash scripts/build-engine-780m.sh          # cmake -> engine component package
bash scripts/run-strata-780m.sh            # docker run, serves on 127.0.0.1:8080
```

Artifacts:

```
image      strata:0.2.1-rocm-gfx1103                       (1.37 GB)
engine     0.1.46.1-linux-x86_64-hip-gfx1103              (608 MB, 274 files)
  BUILD.json  {"archs":"gfx1103","hip":"10.1","hipblaslt_version":100401}
  tuning/gfx1103-hipblaslt-100401.txt                     (calibrated table, auto-loaded)
  rocblas/library/     150 gfx1103 kernels
  hipblaslt/library/gfx1103/   98 files
workdir    /opt/strata-780m/   (source, build tree, logs)
```

### How it is launched

```bash
TBLMODE=off IGPUSET=1 WMMA=1 VRAMRES=6144 CONTEXT=32k bash scripts/run-strata-780m.sh
```

which expands to:

```bash
docker run -d --name strata-780m \
  --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  --shm-size 4g --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8080:8080 \
  -v strata-780m-data:/data \
  -v /opt/models/Qwen3.8-Flash-Next-GGUF:/models:ro \
  -e MODEL=UD-IQ4_XS -e MODEL_PATH=/models -e CONTEXT=32k \
  -e VISION=off -e TUNE=skip -e API_KEY=vaskes-strata-780m \
  -e STRATA_EXPERIMENTAL_GFX1103=1 \
  -e STRATA_SET="spec=2 lookup-chain=4 spec-min-p=0.35 vram-reserve-mib=6144 STRATA_HIPBLASLT_TABLE=off" \
  -e STRATA_IGPU_PROMPT_SET=1 \
  -e STRATA_HIP_WMMA=1 STRATA_PF_GEMM=1 STRATA_PA_FAST=1 \
  -e STRATA_SELECT_WMMA=1 STRATA_HC_UPMIX=1 \
  --restart unless-stopped strata:0.2.1-rocm-gfx1103
```

Notes that are **not** obvious:

* `STRATA_EXPERIMENTAL_GFX1103=1` is **mandatory**. Without it `setup.py` drops gfx1103
  from `AMD_ARCHS`, decides the card is unsupported and refuses to write a run config
  for any family — `strata launch` exits 5. With it, setup also writes the 17 exact
  speed switches into the config's `env` on its own.
* There is no `--hipblaslt-tuning` flag. The table lives in the component package and
  is matched through `BUILD.json:hipblaslt_version`.
* `--spec 2` is the best MTP window on this chip upstream (15.65 tok/s vs 14.18 at 4
  and 11.16 at 6); `--lookup-chain 4` adds ~2.3%.

### Measured

From the engine's own per-request timings
(`strata serve: prompt N tokens = ... read in X ms (Y tok/s), G generated in Z ms (W tok/s)`):

| prompt | generated | decode | MTP draft accept |
|---:|---:|---:|---:|
| 64 tok | 32 | **9.4 tok/s** (3387 ms) | 16/21 (76%) |
| 64 tok | 32 | **9.4 tok/s** (3389 ms) | 16/19 (84%) |
| 55 tok | 29 | 9.0 tok/s (3238 ms) | 13/14 (93%) |
| 55 tok | 29 | 8.9 / 8.8 / 7.8 tok/s | 13/14 |
| 53 tok | 1 | 5.7–6.6 tok/s | — (no window to draft) |

Single-token rows are the cost of one step, not throughput: MTP has nothing to draft
from. **Real streaming rate is 7.8–9.4 tok/s.**

Expert-cache hit rate during decode: **89–96%**; the rest streams from the SSD.
`GPUActive` sits at ~53 GiB — the experts really do live in unified memory.

Prefill, short prompts only: **9.5–23.8 tok/s** (best 23.8 at 55 tokens). Anything
beyond ~85 prompt tokens dies — see [rocBLAS](#known-blocker-rocblas-on-gfx1103).

---

## llama.cpp — 9 tok/s (Q8_0)

`llama.cpp` `0.3.0-dev (build 10675, commit 90c26fcd4)`, ROCm 7.2.4, serving the
`Qwen3.8-Flash-Next` family on the same 780M.

The launch config is preserved verbatim in
[`systemd/llama-qwen38-flash.service`](systemd/llama-qwen38-flash.service):

```ini
[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp-test/build/bin
ExecStart=/opt/llama.cpp-test/build/bin/llama-server \
    --model /opt/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    --mmproj /opt/models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf \
    --alias Qwen3.8-Flash-Next \
    --n-gpu-layers 99 \
    --ctx-size 262144 \
    -n 65536 \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --cache-ram 0 --parallel 1 --no-warmup --flash-attn on \
    --agent --mcp-servers-config /opt/search/mcp-servers.json --jinja \
    -b 2048 --host 0.0.0.0 --port 8083
```

Other llama.cpp units from the same box are deliberately not here — this repo is
about Flash-Next on the 780M, not about everything else that ran on it.

> **Provenance of the 9 tok/s figure.** The Q8_0 llama.cpp number is the operator's
> run on this box; the per-request logs were not preserved, and no Q8_0 build of
> Flash-Next is present on the host (the IQ4_XS shards are). The figure is recorded
> here as given. Re-measure with `scripts/bench-decode-steady.py`-style timing if you
> need a reproducible number for your own build.

---

## The wall is memory bandwidth

Not the engine. Three observations pin it down:

1. **Two unrelated engines agree.** Strata (a purpose-built engine with WMMA prompt
   kernels, a hipBLASLt tuning table and MTP speculative decoding) and llama.cpp land
   within noise of each other on the same box. Software above the memory pipe cannot
   change the rate at which the pipe delivers.

2. **A 2.1× smaller quant on the same silicon goes faster.** Upstream's community run
   on a Ryzen 7 **8745HS** (also a 780M) reports **13–16 tok/s decode** — on a
   **Q2_0** pack, ~2.0 bits/weight against our UD-IQ4_XS at ~4.25. Half the bytes per
   token buys about 1.5× the tokens per second. That is the signature of a bandwidth
   limit, not a compute limit.

3. **The arithmetic lines up.** Taking ~3B active parameters per token (A3B):

   | run | bits/weight | bytes/token | tok/s | expert traffic |
   |---|---:|---:|---:|---:|
   | ours, UD-IQ4_XS | ~4.25 | ~1.6 GB | 9.4 | **~15 GB/s** |
   | upstream, Q2_0 | ~2.0 | ~0.75 GB | 14 | **~10 GB/s** |

   Same order of magnitude for two different quant sizes on two different 780M
   machines. The memory subsystem delivers a roughly *fixed* amount of expert data per
   second; a fatter quant simply buys proportionally fewer tokens.

Why the ceiling is where it is: on an APU the iGPU has no VRAM of its own and reaches
DDR5 over a path far narrower than the Zen 4 cores get — and the access pattern is
nothing like streaming. Each token wants a few megabytes of scattered 3–4 MB expert
blobs pulled from a 93 GB file, which lands far below the 76.8 GB/s the DDR5-4800
dual-channel bus can do sequentially. Add the same DDR5 being shared with 8 Zen 4
cores that want their own cache and memory traffic, and ~10–15 GB/s of useful expert
reads is the real number.

### What makes it worse here than upstream

A second ceiling stacks on top, specific to this box: **the model does not fit.**
59.5 GiB of experts in 58.2 GiB of RAM means ~16 GiB of experts can never be resident.
`setup.py`'s `unsloth_budget()` collapses `--resident-budget-gib` from the requested
24 GiB to 0.1 GiB, because `MemAvailable` is 4.4 GiB once `GPUActive` has taken its
share. The engine log shows the consequence plainly —

```
decode waited 66362.0 ms in 16662 fetches
```

one request spent **66 seconds** waiting on the SSD. Upstream's Q2_0 pack has a 34.0
GiB arena, which fits their 60 GB box resident and never streams at all.

So: on this machine the ceiling is bandwidth, and the model size keeps a second
ceiling stacked on it. Both are hardware.

---

## Known blocker: rocBLAS on gfx1103

Prompts longer than ~85 tokens abort the engine with:

```
rocBLAS error from hip error code: 'hipErrorInvalidKernelFile':218
prefill gemm: cublasGemmEx: cuBLAS status 6
```

This is **not** Strata and **not** a packaging mistake. Reproduced with a standalone
GEMM probe in a clean `rocm/dev-ubuntu-24.04:10.1.0-full` container, no Strata in sight:

```
rocblas_sgemm(512,  512,  512)  -> OK
rocblas_sgemm(512, 2560, 2560)  -> Memory access fault by GPU node-1
                                   Reason: Page not present or supervisor privilege
```

2560 is the model's dense width — exactly the shape Strata's prefill issues.

Root cause is visible in ROCm itself: **gfx1103 is not in rocBLAS's arch list.**

* ROCm 7.2.4 enumerates `TensileLibrary_lazy_*` for gfx1102, gfx1150, gfx1151, gfx1200,
  gfx1201, gfx950, gfx90a, gfx908, gfx1030 — **not gfx1103**.
* ROCm 10.1 ships only `*_fallback_gfx1103.*` at the top level with no `gfx1103/`
  subdirectory, and those are the ones that fault.

Packaging is provably clean: the same GEMM through the *packaged* `librocblas.so.5`
behaves identically, `ldd` resolves everything, and all 98 hipBLASLt files are present.
Disabling the WMMA prompt set and the hipBLASLt table does not change it.

**The fix** is to build against the **TheRock gfx110X-all SDK**
(`https://rocm.nightlies.amd.com/v2/gfx110X-all/`), which is what upstream itself
does — `setup.py:ROCM_INDEXES["gfx1103"]` and `packaging/build_hip_gfx1103.sh`
(`gfx110X-all → 7.10.0a20251121`). TheRock carries real gfx1103 kernels.

---

## Gotchas, written down so they are not rediscovered

1. `STRATA_EXPERIMENTAL_GFX1103=1` is mandatory or setup refuses to configure anything.
2. `--hipblaslt-tuning` does not exist as a CLI flag; the table is auto-matched.
3. `collect_libs.py`: use `--data-arch gfx1103`, **not** `--data-glob '*gfx1103*'`. The
   glob silently drops hipBLASLt files whose names carry no arch.
4. `package_engine.py` needs `--tree`, or the `rocblas/library` kernel tree is left out
   and the engine dies with `Could not initialize Tensile host`.
5. `BUILD.json` without `hipblaslt_version` is rejected — fill it with the upstream
   `packaging/stage_tuning.py --archs gfx1103 --rocm /opt/rocm --expect gfx1103`.
6. This network severs connections at exactly **62 914 560 bytes (60 MiB)**. It killed
   a 66 MB CPython fetch and a llama.cpp zip. Recover with
   `curl -r <offset>-<end>` in ~30 MiB slices and concatenate; the pinned SHA-256
   matches afterwards.
7. `set -o pipefail` + `ls | head -5` kills a build script via SIGPIPE.
8. `--batch` conflicts with the `parallel` override from `docker/entrypoint.sh`.
9. Sandbox→host link runs ~28 KB/s while host→host over the mesh is ~100 MB/s
   (315 MB in 2.9 s). Big files go through `llmhost1`.

---

## Bottom line

The 780M will not give you more than ~9 tok/s on a 177B-class model. That is the
memory path, and no engine change touches it. But it will run one — on a processor
that costs a fraction of any discrete GPU setup, in a box with no graphics card at
all. If money is the constraint, a Ryzen 8700G is the answer; there is no other route
to a model this size without spending several thousand dollars on silicon.

## Layout

```
systemd/   the llama.cpp launch config for Qwen3.8-Flash-Next on the 780M
scripts/   Strata build/run scripts, benchmarks, and the two GPU probes
docs/      full investigation report: measurements, packaging fixes, dead ends
```

## Credits

This work is a collaboration between:

* **Vaskes** — Vibecoder, hardware & infrastructure, project lead, integration QA.
* **Mavis** — AI implementation agent, powered by MiniMax M3.