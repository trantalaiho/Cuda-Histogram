/*
 *
 *
 *  Created on: 6.1.2012
 *      Author: Teemu Rantalaiho (teemu.rantalaiho@helsinki.fi)
 *
 *
 *  Copyright 2011-2012 Teemu Rantalaiho
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *
 *  Compile with:
 *
 *   nvcc -O4 -arch=<your_arch> -I../ test_minimal.cu -o test_minimal
 *
 */

#include "cuda_histogram.h"
#include <assert.h>
#include <stdio.h>

// minimal test - 1 key per input index
struct test_xform {
  __host__ __device__
  void operator() (int* input, int i, int* res_idx, int* res, int nres) const {
    *res_idx++ = input[i];
    *res++ = 1;
  }
};

// Sum-functor to be used for reduction - just a normal sum of two integers
struct test_sumfun {
    __device__ __host__ int operator() (int res1, int res2) const{
        return res1 + res2;
    }
};

// Entrypoint
int main (int argc, char** argv)
{
    // Allocate an array on CPU-side to fill data of 10 indices
    int h_data[] = { 0, 1, 0, 2, 2, 3, 1, 5, 0, 0 };
    // Allocate an array on GPU-memory to hold the input
    int* d_data = NULL;
    cudaMalloc(&d_data, sizeof(h_data));
    assert(d_data); // Probably something wrong with GPU initialization
                    // - check your CUDA runtime and drivers.
    // Copy the input-data to the GPU
    cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);

    // Init the result-array on host-side to zero (we have 6 bins, from 0 to 5):
    int res[6] = { 0 };

    // Create the necessary function objects and run histogram using them
    // 1 result per input index, 6 bins, 10 inputs
    test_xform xform;
    test_sumfun sum;
    callHistogramKernel<histogram_atomic_inc, 1>(d_data, xform, sum, 0, 10, 0, &res[0], 6);

    // Print the result, should be: [ 4, 2, 2, 1, 0, 1 ]
    printf("Results:  [ %d, %d, %d, %d, %d, %d ]\n",
            res[0], res[1], res[2], res[3], res[4], res[5]);
    printf("Expected: [ 4, 2, 2, 1, 0, 1 ]\n");
    // Done: Let OS + GPU-driver+runtime worry about resource-cleanup
    return 0;
}
