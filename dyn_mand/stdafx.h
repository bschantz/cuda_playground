#pragma once

#include "cuda_runtime.h"
#include "cuda_occupancy.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "helper_cuda.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <png.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!\n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }