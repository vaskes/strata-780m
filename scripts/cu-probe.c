#include <hip/hip_runtime.h>
#include <stdio.h>

int main(void) {
    hipDeviceProp_t p;
    hipGetDeviceProperties(&p, 0);
    printf("arch=%s CUs=%d warp=%d clock=%dkHz memBusWidth=%dbit memClock=%dkHz\n",
           p.gcnArchName, p.multiProcessorCount, p.warpSize, p.clockRate,
           p.memoryBusWidth, p.memoryClockRate);
    printf("theoreticalPeakDDR (GB/s) = %.1f\n",
           2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6);
    return 0;
}