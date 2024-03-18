#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>
__global__ void helloFromGpu() {
	printf("hello from GPU!\n");
}

int main(int argc, char *argv[])
{

	helloFromGpu<<<1,10>>> ();		
	cudaDeviceReset();
	return 0;
}
