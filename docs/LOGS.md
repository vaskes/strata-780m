# Logs and build output

Raw logs from the investigation on `llmhost2`. Kept as-is, including the failed runs —
the dead ends are the useful part.

Build / packaging
-----------------
* `pull.log`      — `docker pull rocm/dev-ubuntu-24.04:10.1.0-full`
* `build.log`     — first cmake attempt: `cmake: command not found` (the rocm/dev image is minimal)
* `build2.log`    — `could not find git for clone of strata_llamacpp` (FetchContent wants to re-clone llama.cpp)
* `build3.log`    — engine linked; `package_engine.py` refuses: `BUILD.json has no hipblaslt_version`
* `build4.log`    — `--data-glob` variant; rocBLAS kernels staged but *not* in the package
* `build5.log`    — killed by `set -o pipefail` + `ls | head` (SIGPIPE)
* `build6.log`    — unterminated string: a `"` inside a comment terminated the outer shell string
* `build7.log`    — **package built**, but `hipErrorInvalidKernelFile` at prefill (missing hipblaslt kernels)
* `build8.log`    — `--data-arch gfx1103` + `--tree`: **274 files, 150 rocblas kernels, 608 MB**
* `image.log`     — first image build: cpython download cut at 62 914 560 / 66 913 335 bytes
* `image2.log`    — cpython pre-staged; Terminal built, `strata 0.2.1 (build 0.1.46.1)`
* `image3.log`    — final image `strata:0.2.1-rocm-gfx1103`, 1.37 GB
* `clone.log` / `clone2.log` — git clone from the host (~40 KB/s at the time)

Benchmark runs
--------------
* `bench.log`, `bench2.log`    — 4K/32K prompts; aborted by the rocBLAS fault
* `ceiling.log`, `ceiling2.log` — prompt-size walk; same
* `decode*.log`                — 256-token steady-state attempts; the engine dies partway on the longer runs

Other
-----
* `cpy.log`, `rest.log` — the ranged re-fetch that recovered the 66 MB CPython archive
* `sha.log`             — SHA-256 of the three UD-IQ4_XS shards (matches Strata's pin)
* `eng-head.txt`        — the rocBLAS proof, see below

## The two GPU probes

Both are in `scripts/` and both are how the rocBLAS blocker was pinned down without
guessing.

`scripts/rocblas-probe.c` — one small FP32 GEMM through the packaged rocBLAS:

```
rocblas_create_handle: 0
rocblas_sgemm: 0
hipDeviceSynchronize: 0 (no error)
device: gfx1103  arch 11.0  CUs 6  mem 54.8 GB
```

> **`CUs 6` here, `12` in the README.** `rocminfo` reports 12 compute units for the
> gfx1103 agent (matching the 780M spec), while `hipDeviceProp_t.multiProcessorCount`
> — the field printed above — reports 6. The two ROCm interfaces disagree on this box.
> The probe value is quoted verbatim because this is a verbatim probe log; the README
> uses the spec-matching 12. `scripts/cu-probe.c` prints both.

`scripts/gemm-shapes-probe.cpp` — the same walk over M, which is what Strata's prefill
grows. Compile inside the ROCm image:

```sh
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  -e HIP_PLATFORM=amd -v "$PWD":/w rocm/dev-ubuntu-24.04:10.1.0-full \
  bash -lc "cd /w && hipcc -O2 --offload-arch=gfx1103 -D__HIP_PLATFORM_AMD__ \
            gemm-shapes-probe.cpp -o gp -L/opt/rocm/lib -lrocblas && stdbuf -o0 ./gp"
```

Result:

```
FP32 GEMM by M on the 780M (prefill grows M with the prompt):
Memory access fault by GPU node-1 on address 0x...  Reason: Page not present or supervisor privilege.
```

It faults on the **first** shape, `M=512, N=K=2560`. A plain `512³` sgemm passes.
2560 is the model's dense width. gfx1103 is absent from rocBLAS's arch list in both
ROCm 7.2.4 and 10.1 — see the README.