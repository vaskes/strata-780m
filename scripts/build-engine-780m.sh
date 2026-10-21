#!/usr/bin/env bash
# =============================================================================
# Strata HIP engine for gfx1103 (Radeon 780M, Ryzen 7 8700G) — build on llmhost2
# =============================================================================
# Follows docs/RADEON_780M.md from Niko1221/Strata, section 2 "Toolchain and build":
#   ROCm 10.1, HIP 7.16.26385, hipBLASLt 1.4.1, clang 24 — all inside the docker image
#   rocm/dev-ubuntu-24.04:10.1.0-full (the host's /opt/rocm 7.2.4 is NOT usable: its
#   hipBLASLt 1.2.0 has no gfx1103 tuning table, and 7.2 predates the port).
#
# Output: an engine-component package folder that docker/build.sh --flavor hip
# can bake into the runtime image.
# =============================================================================

set -euo pipefail

WORK=/opt/strata-780m
SRC="$WORK/strata"
# cmake is pointed at /work inside the container; /work is $WORK, so the build tree lands at $WORK/build-780m
BUILD="$WORK/build-780m"
IMG=rocm/dev-ubuntu-24.04:10.1.0-full
STAGE="$WORK/stage"
ENGINE="$WORK/engine-pkg"
JOBS="$(nproc)"

log() { printf '\n=== %s\n' "$*"; }

# --- 0. preflight -------------------------------------------------------------
log "preflight"
[ -d "$SRC/third_party/llama.cpp/ggml" ] || { echo "FATAL: no third_party/llama.cpp/ggml"; exit 1; }
echo "source:   $SRC"
echo "llama.cpp pinned commit: 3cf03257f219afbe7334045ff7c6a06ac68c627d (verified in staging)"
echo "jobs:     $JOBS"
df -h /opt | tail -1
free -g | head -2

# --- 1. build the engine inside the ROCm 10.1 image --------------------------
log "cmake configure (HIP, gfx1103)"
# do NOT wipe an existing build tree: ninja is incremental and a full HIP build is slow
docker run --rm \
  -v "$SRC":/src -v "$WORK":/work \
  -w /src \
  "$IMG" \
  bash -lc "
set -euo pipefail
export ROCM=/opt/rocm
# the rocm/dev image ships neither cmake nor ninja
if ! command -v cmake >/dev/null || ! command -v ninja >/dev/null || ! command -v git >/dev/null; then
  echo '--- installing cmake/ninja/build tools/git ---'
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    cmake ninja-build build-essential python3-pip ca-certificates git libcurl4-openssl-dev
fi
cmake --version | head -1
ninja --version
git --version
# llama.cpp is already in the tree at the pinned commit; point FetchContent at it so
# CMake does not try to clone it again (the host's GitHub is slow and git may be absent)
cmake -S /src -B /work/build-780m -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1103 \
  -DCMAKE_HIP_COMPILER=\$ROCM/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=\$ROCM \
  -DCMAKE_PREFIX_PATH=\$ROCM \
  -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=/src/third_party/llama.cpp \
  '-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm --rocm-device-lib-path=/opt/rocm/lib/llvm/amdgcn/bitcode' 2>&1 | tail -40
echo '--- configure done, building ---'
cmake --build /work/build-780m -j $JOBS 2>&1 | tail -30
echo '--- artifacts ---'
ls -la /work/build-780m/strata /work/build-780m/libstrata_kq.so 2>&1 || true
"

log "engine binary built?"
ls -la "$BUILD/strata" || { echo "FATAL: no engine binary"; exit 1; }

# --- 2. package it as an engine component ------------------------------------
log "package engine component"
rm -rf "$STAGE" "$ENGINE"
mkdir -p "$STAGE"

cp "$BUILD/strata" "$STAGE/"
[ -f "$BUILD/libstrata_kq.so" ] && cp "$BUILD/libstrata_kq.so" "$STAGE/" || true

VER="$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' "$SRC/CMakeLists.txt" | head -1)"
echo "{\"version\": \"$VER\", \"source\": \"docker build gfx1103\", \"archs\": \"gfx1103\", \"hip\": \"10.1\", \"portable\": true}" > "$STAGE/BUILD.json"

# collect the shared libraries the engine actually needs (ROCm user-space lives
# in the image, but the package must carry what setup expects).
# The --data/--data-glob pair matters: rocBLAS reads its TensileLibrary kernels at
# startup from rocblas/library/ next to the .so, and a gfx1103 engine only needs the
# gfx1103 rows of it (the whole tree is 809 MB).
# Collect the shared libraries the engine needs, plus the rocBLAS/hipBLASLt kernel trees.
# This mirrors packaging/build_hip_gfx1103.sh (upstream's own gfx1103 recipe): use --data-arch,
# NOT --data-glob. A '*gfx1103*' glob silently drops hipblaslt/library files whose names carry
# no arch, and the engine then dies at prefill with
#   rocBLAS error: hipErrorInvalidKernelFile / prefill gemm: cuBLAS status 6 (verified).
docker run --rm -v "$SRC":/src -v "$WORK":/work -v "$STAGE":/stage -w /src \
  "$IMG" bash -lc "
set -euo pipefail
python3 packaging/collect_libs.py --binary /work/build-780m/strata --search /opt/rocm/lib --out /stage \
  --also 'libamd_comgr.so*' --also 'libhiprtc*.so*' \
  --data /opt/rocm/lib/rocblas/library=rocblas/library \
  --data /opt/rocm/lib/hipblaslt/library=hipblaslt/library \
  --data-arch gfx1103 \
  2>&1 | tail -6
echo '--- kernel trees staged ---'
ls /stage/rocblas/library/ 2>/dev/null | wc -l
ls /stage/hipblaslt/library/ 2>/dev/null | wc -l
test -d /stage/rocblas/library || { echo 'FATAL: rocblas kernels not staged'; exit 1; }
"

echo "--- staged files ---"
ls -la "$STAGE" | head -20
du -sh "$STAGE"

log "stage_tuning.py (records hipblaslt_version + copies the gfx1103 table)"
# This is what package_engine.py demands: it reads hipblaslt-version.h from the ROCm tree,
# writes hipblaslt_version into BUILD.json and copies tools/hip/gfx1103-hipblaslt-<version>.txt.
# Our ROCm 10.1 ships hipBLASLt 1.4.1 (= 100401), which is exactly the table docs/RADEON_780M.md
# measured with, so --expect gfx1103 is safe and turns a mismatch into a hard error.
docker run --rm -v "$SRC":/src -v "$STAGE":/stage -w /src \
  "$IMG" bash -lc "
set -euo pipefail
python3 packaging/stage_tuning.py --stage /stage --archs gfx1103 --rocm /opt/rocm --expect gfx1103
echo '--- tuning staged ---'
ls -la /stage/tuning/ 2>/dev/null || echo '(no tuning dir)'
cat /stage/BUILD.json
"

log "package_engine.py"
mkdir -p "$ENGINE"
docker run --rm -v "$SRC":/src -v "$WORK":/work -v "$STAGE":/stage -v "$ENGINE":/out -w /src \
  "$IMG" bash -lc "
set -euo pipefail
# --tree: a HIP package ships its rocblas/library kernel tree next to the .so files and the
# engine reads TensileLibrary at startup. Without it the engine aborts with
# 'rocBLAS error: Could not initialize Tensile host' (verified on the first run).
python3 packaging/package_engine.py \
  --src /stage --backend hip --version '$VER' --out /out \
  --arch gfx1103 \
  --variant gfx1103 --platform linux-x86_64 \
  --build-id 'llmhost2-780m-$VER' \
  --tree \
  --include SYSTEM-LIBS.txt \
  --include 'libstrata_kq*'
"

echo "--- engine package ---"
find "$ENGINE" -maxdepth 2 -type f | head -8
du -sh "$ENGINE"
echo "rocblas kernels in package: $(ls "$ENGINE"/*/rocblas/library 2>/dev/null | wc -l)"

log "verify package"
docker run --rm -v "$SRC":/src -v "$ENGINE":/engine -w /src "$IMG" bash -lc "
set -euo pipefail
for d in /engine/*/; do
  python3 packaging/package_engine.py --verify \"\$d\"
  python3 packaging/binary_deps.py \"\$d\" 2>&1 | tail -25
done
"

log "DONE — engine package at $ENGINE"