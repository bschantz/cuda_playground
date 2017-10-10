#pragma once

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include "cuda_runtime.h"
#include "cuda_occupancy.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "device_atomic_functions.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>

#define CudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!\n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }