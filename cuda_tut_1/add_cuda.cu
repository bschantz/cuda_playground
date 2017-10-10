// non_cuda_1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__
void add(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i+=stride)
	{
		y[i] = x[i] + y[i];
	}
}

std::vector<int> * getComputeDevices(void)
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	int device;
	std::vector<int> *ret = new std::vector<int>(deviceCount);
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		std::cout << "Device " << deviceProp.name << " is available." << std::endl;
		std::cout << "Compute version " << deviceProp.major << "." << deviceProp.minor << std::endl;
		std::cout << "Got " << deviceProp.concurrentManagedAccess << " in deviceProp.concurrentManaged access." << std::endl;
		ret->push_back(device);
	}

	return ret;
}

int main(void)
{
	std::vector<int> * devices = getComputeDevices();
	int device;
	if (!devices->empty()) {
		device = devices->front();
		cudaSetDevice(device);
	}
	else 
	{
		std::cout << "No compute devices found." << std::endl;
		delete devices;
		return -1;
	}

	int N = 1 << 22;

	// Allocate unified memory -- accessible from CPU or GPU
	float *x, *y;
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	int blockSize = 256;
	int numBlocks = N + blockSize - 1 / blockSize;
	add <<<numBlocks, blockSize>>>(N, x, y);

	cudaDeviceSynchronize();

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
	{
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}

	std::cout << "Max error: " << maxError << std::endl;

	cudaFree(x);
	cudaFree(y);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::cerr << "cudaDeviceReset failed!" << std::endl;
		return 1;
	}


	delete devices;

	return 0;
}


