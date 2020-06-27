#ifndef _PARAM_H_
#define _PARAM_H_

// Required includes

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jni.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <float.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

// Include JNI interfaces

#include "jni/weka_filters_supervised_attribute_CAIMGPU.h"

using namespace thrust;
using namespace std;

// Number of threads per block at evaluation kernels
#define THREADS_EVAL_BLOCK 128
#define MAX_CLASSES 48

#endif
