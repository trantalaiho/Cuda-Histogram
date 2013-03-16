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
 *   Compile with:
 *  
 *    nvcc -O4 -arch=<your_arch> -I../ test_simple.cu -o test_simple
 *
 */

#define TESTMAXIDX   256
#define TEST_IS_POW2 1
#define TEST_SIZE (25 * 100 * 1000 )   // 2.5 million inputs

#include "cuda_histogram.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// NOTE: Code assumes sizeof(unsigned int) == 4 - if not, typedef myUint32 accordingly
typedef unsigned int myuint32;



// Do the following:
// Consider every 8-bit input as a 8-bit luminance value and histogram accordingly
// Inputs come in groups of 128 bits, in order to run faster on Fermi
struct test_xform
{
  __host__ __device__
  void operator() (uint4* input, int i, int* result_index, int* results, int nresults) const {
    uint4 idata = input[i];
    unsigned char * data = (unsigned char *)&idata;
#pragma unroll
    for (int resIdx = 0; resIdx < 16; resIdx++){
        *result_index++ = *data++;
        *results++ = 1;
    }
  }
};

// Sum-functor to be used for reduction - just a normal sum of two integers
struct test_sumfun {
    __device__ __host__ int operator() (int res1, int res2) const{
        return res1 + res2;
    }
};

// Helper function to print the bins of the result histogram
static void printres (int* res, int nres, const char* descr)
{
    if (descr)
        printf("\n%s:\n", descr);
    printf("vals = [\n");
    for (int i = 0; i < nres; i++)
        printf("%d, ", res[i]);
    printf("]\n");
}

// Get Gaussian random number with given mean value and standard deviation
static float gaussianRnd(float mean, float stdDev)
{
    static bool prevRes = false;
    static float prev = 0.0f;
    float result;
    if (prevRes){
        result = prev * stdDev + mean;
        prevRes = false;
    } else {
        // Typical box-Muller
        float u1 = ((float)rand())/((float)RAND_MAX);
        float u2 = ((float)rand())/((float)RAND_MAX);
        if (u1 <= 0.0f || u1 > 1.0f) u1 = 1.0f;
        float r = sqrt(-2.0f*log(u1));
        float a = 2.0f * /*PI*/ 3.141592654 * u2;
        result = (r * cos(a)) * stdDev + mean;
        prev = r * sin(a);
        prevRes = true;
    }
    return result;
}

// Mirror into [0,1]
static float mirrorToUnit(float x)
{
    int z;
    float z2;
    float y = modf(fabs(x), &z2);
    z = (int)z2;
    if ((z & 0x1) != 0) y = 1.0f - y;
    return y;
}

// Fill input-array
static void fillInput(myuint32 * data, int nPixels)
{
    for (int i = 0; i < nPixels; i++){
        // Let the channels be gaussian distributions with the following parameters:
        // r - mean: 0.2, std-dev: 0.1
        // g - mean: 0.6, std-dev: 0.4
        // b - mean: 0.9, std-dev: 0.1
        // a - mean: 0.5, std-dev: 0.5
        // What is the form of the resulting histogram distribution (a*(r+g+b))? Gaussian?
        int r = (int)(255.99f * mirrorToUnit(gaussianRnd(0.2, 0.1)));
        int g = (int)(255.99f * mirrorToUnit(gaussianRnd(0.6, 0.4)));
        int b = (int)(255.99f * mirrorToUnit(gaussianRnd(0.9, 0.1)));
        int a = (int)(255.99f * mirrorToUnit(gaussianRnd(0.5, 0.5)));
        unsigned int result = (a << 24) | (r << 16) | (g << 8) | (b);
        *data++ = result;
    }
}

// Entrypoint
int main (int argc, char** argv)
{
    if (sizeof(myuint32) != 4){
        printf("Sorry - sizeof(myuint32) == %d, needs to be 4, please edit typedef of myuint32.\n", sizeof(myuint32));
        return 1;
    }
    // Seed rnd generator
    srand(3);

    // Allocate an array on CPU-side to fill data
    myuint32* h_data = (myuint32*)malloc(TEST_SIZE * sizeof(myuint32));
    assert(h_data);

    // Fill the input using subroutine
    fillInput(h_data, TEST_SIZE);

    // Allocate an array on GPU-memory to hold the input
    myuint32* d_data = NULL;
    cudaMalloc(&d_data, TEST_SIZE * sizeof(myuint32));
    assert(d_data); // Probably something wrong with GPU initialization - check your CUDA runtime and drivers.

    // Copy the input-data to the GPU
    cudaMemcpy(d_data, h_data, TEST_SIZE * sizeof(myuint32), cudaMemcpyHostToDevice);

    // Init the result-array on host-side to zero:
    int results[TESTMAXIDX] = { 0 };

    // Create the necessary function objects and run histogram using them - 16 results per input index:
    test_xform xform;
    test_sumfun sum;
    callHistogramKernel<histogram_atomic_add, 16>((uint4*)d_data, xform, sum, 0, TEST_SIZE/4, 0, &results[0], TESTMAXIDX);
    printres(results, TESTMAXIDX, "Results");

    // Confirm results using CPU:
    int h_results[TESTMAXIDX] = { 0 };
    for (int i = 0; i < TEST_SIZE / 4;i++){
        int res[16];
        int indices[16];
        xform((uint4*)h_data, i, &indices[0], &res[0], 8);
        for (int resid = 0; resid < 16; resid++)
            h_results[indices[resid]] = sum(h_results[indices[resid]], res[resid]);
    }
    for (int binid = 0; binid < TESTMAXIDX; binid++)
        if (h_results[binid] != results[binid]){
            printf("Host results differ from GPU results by more than relative 1e-4f: absolute err[%d] = %f\n",
                binid, fabs(h_results[binid] - results[binid]));
            printres(h_results, TESTMAXIDX, "CPU-Results");
            return 2;
        }
    // Done: Let OS + GPU-driver+runtime worry about resource-cleanup
    return 0;
}

