// Does rocBLAS break at large M on gfx1103? Strata's prefill does one big GEMM per chunk
// (M = prompt chunk), so if this fails at some M the same kernel is what kills the engine.
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdio.h>

static void try_sgemm(rocblas_handle h, int M, int N, int K) {
    size_t bytes = (size_t)M * K * sizeof(float);
    size_t cbytes = (size_t)M * N * sizeof(float);
    float *A, *B, *C;
    if (hipMalloc((void **)&A, bytes) != hipSuccess) { printf("alloc A fail\n"); return; }
    hipMalloc((void **)&B, (size_t)K * N * sizeof(float));
    hipMalloc((void **)&C, cbytes);
    hipMemset(A, 0, bytes); hipMemset(B, 0, (size_t)K * N * sizeof(float)); hipMemset(C, 0, cbytes);

    const float alpha = 1.0f, beta = 0.0f;
    rocblas_status st = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                                      M, N, K, &alpha, A, K, B, N, &beta, C, N);
    hipError_t he = hipDeviceSynchronize();
    printf("  M=%-6d N=%-5d K=%-5d fp32: rocblas %2d  sync %2d  %s\n",
           M, N, K, (int)st, (int)he,
           (st == rocblas_status_success && he == hipSuccess) ? "OK" : "<<< FAIL");
    hipFree(A); hipFree(B); hipFree(C);
}

int main(void) {
    rocblas_handle h;
    if (rocblas_create_handle(&h) != rocblas_status_success) { printf("handle failed\n"); return 1; }
    printf("FP32 GEMM by M on the 780M (prefill grows M with the prompt):\n");
    int Ms[] = {512, 2048, 4096, 8192, 16384};
    for (unsigned i = 0; i < sizeof(Ms) / sizeof(Ms[0]); i++)
        try_sgemm(h, Ms[i], 2560, 2560);
    return 0;
}