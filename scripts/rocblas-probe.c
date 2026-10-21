// Minimal rocBLAS GEMM on gfx1103: does ROCm 10.1's rocBLAS work on a Radeon 780M at all?
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdio.h>

int main(void) {
    const int n = 512;
    size_t bytes = (size_t)n * n * sizeof(float);

    rocblas_handle h;
    rocblas_status st = rocblas_create_handle(&h);
    printf("rocblas_create_handle: %d\n", (int)st);
    if (st != rocblas_status_success) return 1;

    float *A = 0, *B = 0, *C = 0;
    hipMalloc((void**)&A, bytes); hipMalloc((void**)&B, bytes); hipMalloc((void**)&C, bytes);
    hipMemset(A, 0, bytes); hipMemset(B, 0, bytes); hipMemset(C, 0, bytes);

    const float alpha = 1.0f, beta = 0.0f;
    st = rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none,
                       n, n, n, &alpha, A, n, B, n, &beta, C, n);
    printf("rocblas_sgemm: %d\n", (int)st);

    hipError_t he = hipDeviceSynchronize();
    printf("hipDeviceSynchronize: %d (%s)\n", (int)he, hipGetErrorString(he));

    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("device: %s  arch %d.%d  CUs %d  mem %.1f GB\n",
           p.gcnArchName, p.major, p.minor, p.multiProcessorCount,
           (double)p.totalGlobalMem / 1e9);
    return st == rocblas_status_success && he == hipSuccess ? 0 : 2;
}