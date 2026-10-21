#!/bin/bash
# Does the PACKAGED rocBLAS (inside the Strata engine component) work on gfx1103,
# or is it the packaging and not ROCm? Run inside the strata image with the probe binary.
set -u
PKG=$(ls -d /opt/strata/components/engine/*/ | head -1)
echo "package: $PKG"
echo "kernels in package: $(ls "$PKG/rocblas/library" 2>/dev/null | wc -l)"
export LD_LIBRARY_PATH="$PKG"
echo "--- ldd of librocblas as the engine sees it ---"
ldd "$PKG/librocblas.so.5" 2>&1 | grep -i "not found" || echo "all libs resolve"
echo "--- probe against the packaged rocBLAS ---"
cd /w
./rocblas-probe
echo "exit=$?"
echo "--- does rocBLAS look somewhere else? ---"
strings "$PKG/librocblas.so.5" 2>/dev/null | grep -iE "TensileLibrary|rocblas/library|ROCBLAS_TENSLIB" | sort -u | head -10