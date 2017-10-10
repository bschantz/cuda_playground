// prime.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

// __global__ void atkin_sieve(int*);
__global__ void atkin_pass2(int*);
__global__ void atkin_pass1(int*);
__global__ void atkin_pass3(int*);
__global__ void atkin_pass4(int*);

__device__ int test_limit;
__device__ long device_max_prime;

std::vector<int> * getComputeDevices(void);

const unsigned long MAX = 15500000;
long max_prime;

int main(int argc, char **argv)
{
	if (argc == 1) {
		max_prime = (long)(*argv[0]);
	} else if (argc == 0) {
		max_prime = MAX;
	} else {
		std::cout << "Usage: cuda_prime [max_prime]" << std::endl;
		exit(-1);
	}
	std::vector<int> * devices = getComputeDevices();
	int device;
	if (!devices->empty()) {
		device = devices->front();
		checkCudaErrors(cudaSetDevice(device));
	}
	else
	{
		std::cout << "No compute devices found." << std::endl;
		delete devices;
		return -1;
	}

	size_t numBytes = max_prime * sizeof(int);
	// allocate host memory
	int *result;

	// allocate device memory
	checkCudaErrors(cudaMallocManaged(&result, numBytes));
	memset(result, 0, numBytes);

	// calc upper bound of search indices
	int host_limit = (int)std::ceil(std::sqrtf(MAX));
	checkCudaErrors(cudaMemcpyToSymbol(test_limit, &host_limit, sizeof(int)));

	// copy max prime parameter
	checkCudaErrors(cudaMemcpyToSymbol(device_max_prime, &max_prime, sizeof(long)));

	// copy mem to device
	// checkCudaErrors(cudaMemcpy(d_result, result, numBytes, cudaMemcpyHostToDevice));

	// launch kernel
	dim3 blocks(256, 256);
	dim3 threads(16, 16);
	cudaStream_t stream1;
	cudaStreamCreate(&stream1);
	atkin_pass1<<<blocks, threads, 0, stream1>>>(result);

	cudaStream_t stream2;
	cudaStreamCreate(&stream2);
	atkin_pass2<<<blocks, threads, 0, stream2>>>(result);

	cudaStream_t stream3;
	cudaStreamCreate(&stream3);
	atkin_pass3<<<blocks, threads, 0, stream3>>>(result);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// checkCudaErrors(cudaMemcpy(result, d_result, numBytes, cudaMemcpyDeviceToHost));

	result[2] = 1;
	result[3] = 1;
	result[5] = 1;

	atkin_pass4<<<256, 256>>>(result);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Display prime
	long pi = 0;
	for (int i = 2; i < max_prime; i++) {
		if (result[i]) {
			pi++;
			printf("%d\n", i);
		}
	}

	checkCudaErrors(cudaFree(result));

	cudaDeviceReset();

	return 0;
}

__global__ void atkin_pass1(int* result)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_x = blockDim.x * gridDim.x;

	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_y = blockDim.y * gridDim.y;

	for(int x = index_x; x < test_limit; x += stride_x) {
		for (int y = index_y; y < test_limit; y += stride_y) {
			int k = 4 * x * x + y * y;
			if (k < device_max_prime && (k % 12 == 1 || k % 12 == 5) && k % 5 != 0) {
				// make this:
				// result[k] = !result[k];
				// atomic
				atomicXor(result + k, 1);
			}
		}
	}
}

__global__ void atkin_pass2(int* result)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_x = blockDim.x * gridDim.x;

	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_y = blockDim.y * gridDim.y;

	for (int x = index_x; x < test_limit; x += stride_x) {
		for (int y = index_y; y < test_limit; y += stride_y) {
			int k = 3 * x * x + y * y;
			if (k < device_max_prime && (k % 12 == 7) && k % 5 != 0) {
				atomicXor(result + k, 1);
			}
		}
	}
}

__global__ void atkin_pass3(int* result)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_x = blockDim.x * gridDim.x;

	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_y = blockDim.y * gridDim.y;

	for (int x = index_x; x < test_limit; x += stride_x) {
		for (int y = index_y; y < test_limit; y += stride_y) {
			int k = 3 * x * x - y * y;
			if (k < device_max_prime && x > y && (k % 12 == 11) && (k % 5 != 0)) {
				atomicXor(result + k, 1);
			}
		}
	}
}

__global__ void atkin_pass4(int* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (unsigned int n = index + 7; n <= test_limit; n += stride) {
		if (result[n]) {
			int n2 = n * n;
			for (int k = n2; k < MAX; k += n2) {
				result[k] = 0;
			}
		}
	}
}

std::vector<int> * getComputeDevices(void)
{
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));

	int device;
	std::vector<int> *ret = new std::vector<int>(deviceCount);
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));
		std::cout << "Device " << deviceProp.name << " is available." << std::endl;
		std::cout << "Compute version " << deviceProp.major << "." << deviceProp.minor << std::endl;
		std::cout << "Multiprocessor count: " << deviceProp.multiProcessorCount << std::endl;
		ret->push_back(device);
	}

	return ret;
}

