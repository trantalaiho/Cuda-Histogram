/*
 *
 *
 *  Created on: 20.1.2012
 *      Author: Teemu Rantalaiho (teemu.rantalaiho@helsinki.fi)
 *
 *
 *  Copyright 2012 Teemu Rantalaiho
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
 *   nvcc -O4 -arch=<your_arch> -I../ test_perf.cu -o test_perf
 *
 *  Optionally include thrust codepath with -DTHRUST
 *
 *
 */

#define TEST_SIZE (625 * 10 * 1000)    // Multiply this by 16 to get size in bytes

// Seems like the other codepath is always better - so disable this one by setting limit to zero
#define SMALL_HISTO_LIMIT   0 // 1000 * 1000 // If there are less than 16million inputs (~4096x4096)
                                        // then do just one 32-bit word at a time (instead of 4)

#define NRUNS 100            // Repeat 20 times => 20 Gigakeys in total (16 keys per entry)
                            // Or for 4 inputs per entry -> 5 Gigakeys

// Define the histogram sizes to go through
const int checkNBins[] =
{
    1,2,3,4,
    8,12,16,20,24,28,32,
    40, 48, 56, 64,
    80, 96, 112, 128,
    160, 192, 224, 256,
    320, 384, 448, 512,
    640, 768, 896, 1024,
    1280, 1536, 1792, 2048,
    2560, 3072, 3584, 4096,
    6144, 8192, 10240, 16384,
    32768, 65536, 131072
};

#ifdef THRUST
#define ENABLE_THRUST   1   // Enable thrust-based version also (xform-sort_by_key-reduce_by_key)
#else
#define ENABLE_THRUST   0
#endif


#include "cuda_histogram.h"
/*#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>*/

#if ENABLE_THRUST
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/inner_product.h>
#endif

#include <assert.h>
#include <stdio.h>


// Always return 1 -> normal histogram - each sample has same weight
struct test_xform16
{
  __host__ __device__
  void operator() (uint4* input, int i, int* result_index, int* results, int nresults) const {
#if __CUDA_ARCH__ >= 200
    uint4 idata = input[i];
    #pragma unroll
    for (int resIdx = 0; resIdx < 4; resIdx++)
    {
        unsigned int data = ((unsigned int*)(&idata))[resIdx];
        *result_index++ = ((data >> 24));
        *result_index++ = ((data >> 16) & 0xFF);
        *result_index++ = ((data >>  8) & 0xFF);
        *result_index++ = ((data >>  0) & 0xFF);
        *results++ = 1;
        *results++ = 1;
        *results++ = 1;
        *results++ = 1;
    }
#else
    unsigned int * ptr = (unsigned int *)input;
    ptr += i;
#pragma unroll
    for (int read = 0; read < 4; read++)
    {
        unsigned int data = *ptr;
        ptr += TEST_SIZE;
        *result_index++ = ((data >> 24));
        *result_index++ = ((data >> 16) & 0xFF);
        *result_index++ = ((data >>  8) & 0xFF);
        *result_index++ = ((data >>  0) & 0xFF);
        *results++ = 1;
        *results++ = 1;
        *results++ = 1;
        *results++ = 1;
    }
#endif
  }
};

// Always return 1 -> normal histogram - each sample has same weight
struct test_xform4
{
  __host__ __device__
  void operator() (uint4* input, int i, int* result_index, int* results, int nresults) const {
#if __CUDA_ARCH__ >= 200
    uint4 idata = input[i];
    #pragma unroll
    for (int resIdx = 0; resIdx < 4; resIdx++)
    {
        unsigned int data = ((unsigned int*)(&idata))[resIdx];
        *result_index++ = data;
        *results++ = 1;
    }
#else
    unsigned int * ptr = (unsigned int *)input;
    ptr += i;
#pragma unroll
    for (int read = 0; read < 4; read++)
    {
        unsigned int data = *ptr;
        ptr += TEST_SIZE;
        *result_index++ = data;
        *results++ = 1;
    }
#endif
  }
};


struct test_xform4small
{
  __host__ __device__
  void operator() (unsigned int* input, int i, int* result_index, int* results, int nresults) const {
    unsigned int data = input[i];
    *result_index++ = ((data >> 24));
    *result_index++ = ((data >> 16) & 0xFF);
    *result_index++ = ((data >>  8) & 0xFF);
    *result_index++ = ((data >>  0) & 0xFF);
    *results++ = 1;
    *results++ = 1;
    *results++ = 1;
    *results++ = 1;
  }
};

struct test_xformsmall
{
  __host__ __device__
  void operator() (unsigned int* input, int i, int* result_index, int* results, int nresults) const {
    unsigned int data = input[i];
    *result_index++ = data;
    *results++ = 1;
  }
};




struct test_sumfun2 {
  __device__ __host__
  int operator() (int res1, int res2) const{
    return res1 + res2;
  }
};


static void printres (int* res, int nres, const char* descr)
{
    if (descr)
        printf("\n%s:\n", descr);
    printf("vals = [ ");
    for (int i = 0; i < nres; i++)
        printf("(%d), ", res[i]);
    printf("]\n");
}

static void testHistogramParam(uint4* INPUT, uint4* hostINPUT, int index_0, int index_1, bool print, bool cpurun, void* tmpBuffer, int nIndices)
{
  int srun;
  int nruns = NRUNS;
  int* tmpres = (int*)malloc(sizeof(int) * nIndices);
  int zero = 0;

  test_xform16 transformFunByte;
  test_xform4 transformFunInt;
  test_sumfun2 sumFun;

  int* gpures;
  cudaMalloc(&gpures, sizeof(int) * nIndices);
  cudaMemset(gpures, 0, sizeof(int) * nIndices);

  for (srun = 0; srun < nruns; srun++)
  {
    if (print)
        printf("\nTest reduce_by_key:\n\n");
    memset(tmpres, 0, sizeof(int) * nIndices);
    if (cpurun)
    {
        if (nIndices <= 256)
        {
            // 16 results per input
            for (int i = index_0; i < index_1; i++) {
              int index[16];
              int tmp[16];
              transformFunByte(hostINPUT, i, &index[0], &tmp[0], 1);
              for (int tmpi = 0; tmpi < 16; tmpi++){
#if 0 // Check inputs...
                  assert(index[tmpi] >= 0);
                  assert(index[tmpi] < nIndices);
#endif
                  tmpres[index[tmpi]] = sumFun(tmpres[index[tmpi]], tmp[tmpi]);
              }
            }
        }
        else
        {
            // 4 results per input
            for (int i = index_0; i < index_1; i++) {
              int index[4];
              int tmp[4];
              transformFunInt(hostINPUT, i, &index[0], &tmp[0], 1);
              for (int tmpi = 0; tmpi < 4; tmpi++)
                  tmpres[index[tmpi]] = sumFun(tmpres[index[tmpi]], tmp[tmpi]);
            }
        }
    }
    if (print && cpurun)
    {
      printres(tmpres, nIndices, "CPU results:");
    }


    if (!cpurun)
    {
#       if (TEST_SIZE <= SMALL_HISTO_LIMIT)
        test_xform4small xformByteSmall;
        test_xformsmall xformIntSmall;
        unsigned int * newInput = (unsigned int *)INPUT;
        if (nIndices <= 256)
            callHistogramKernel<histogram_atomic_inc, 4>(newInput, xformByteSmall, sumFun, 4*index_0, 4*index_1, zero, gpures, nIndices, true, 0, tmpBuffer);
        else
            callHistogramKernel<histogram_atomic_inc, 1>(newInput, xformIntSmall,  sumFun, 4*index_0, 4*index_1, zero, gpures, nIndices, true, 0, tmpBuffer);
#else
        if (nIndices <= 256)
            callHistogramKernel<histogram_atomic_inc, 16>(INPUT, transformFunByte, sumFun, index_0, index_1, zero, gpures, nIndices, true, 0, tmpBuffer);
        else
            callHistogramKernel<histogram_atomic_inc, 4>(INPUT, transformFunInt, sumFun, index_0, index_1, zero, gpures, nIndices, true, 0, tmpBuffer);
#endif
    }
    if (print && (!cpurun))
    {
      printres(tmpres, nIndices, "GPU results:");
    }
    print = false;
  }
  cudaMemcpy(tmpres, gpures, sizeof(int) * nIndices, cudaMemcpyDeviceToHost);
  free(tmpres);
}

#if ENABLE_THRUST
static void testHistogramParamThrust(unsigned char* INPUT, int index_0, int index_1, bool print, int nIndex)
{
  int N = index_1 - index_0;
  thrust::device_vector<int> vals_out(nIndex);
  thrust::host_vector<int> h_vals_out(nIndex);
  //thrust::device_vector<int> keys(N);
  thrust::device_ptr<unsigned char> keys(INPUT);
  // Sort the data
  thrust::sort(keys, keys + N);
  // And reduce by key - histogram complete
#if 0
  // Note: This codepath is somewhat slow
  test_sumfun2 mysumfun;
  thrust::device_vector<int> keys_out(nIndex);
  thrust::equal_to<int> binary_pred;
  thrust::reduce_by_key(keys, keys + N, thrust::make_constant_iterator(1), keys_out.begin(), vals_out.begin(), binary_pred, mysumfun);
#else
  // This is taken from the thrust histogram example
  thrust::counting_iterator<int> search_begin(0);
  // Find where are the upper bounds of consecutive keys as indices (ie. partition function)
  thrust::upper_bound(keys, keys + N,
                      search_begin, search_begin + nIndex,
                      vals_out.begin());
// compute the histogram by taking differences of the partition function (cumulative histogram)
  thrust::adjacent_difference(vals_out.begin(), vals_out.end(),
                              vals_out.begin());
#endif
  h_vals_out = vals_out;
  if (print)
      printres(&h_vals_out[0], nIndex, "Thrust results");
}

static void testHistogramParamThrustInt(int* INPUT, int index_0, int index_1, bool print, int nIndex)
{
  int N = index_1 - index_0;
  thrust::device_vector<int> vals_out(nIndex);
  thrust::host_vector<int> h_vals_out(nIndex);
  //thrust::device_vector<int> keys(N);
  thrust::device_ptr<int> keys(INPUT);
  // Sort the data
  thrust::sort(keys, keys + N);
  // And reduce by key - histogram complete
#if 0
  // Note: This codepath is somewhat slow
  test_sumfun2 mysumfun;
  thrust::device_vector<int> keys_out(nIndex);
  thrust::equal_to<int> binary_pred;
  thrust::reduce_by_key(keys, keys + N, thrust::make_constant_iterator(1), keys_out.begin(), vals_out.begin(), binary_pred, mysumfun);
#else
  // This is taken from the thrust histogram example
  thrust::counting_iterator<int> search_begin(0);
  // Find where are the upper bounds of consecutive keys as indices (ie. partition function)
  thrust::upper_bound(keys, keys + N,
                      search_begin, search_begin + nIndex,
                      vals_out.begin());
// compute the histogram by taking differences of the partition function (cumulative histogram)
  thrust::adjacent_difference(vals_out.begin(), vals_out.end(),
                              vals_out.begin());
#endif
  h_vals_out = vals_out;
  if (print)
      printres(&h_vals_out[0], nIndex, "Thrust results");
}

#endif

void printUsage(void)
{
  printf("\n");
  printf("# Test order independent reduce-by-key / histogram algorithm\n#\n");
  printf("# By default this runs on custom algorithm on the GPU, with lots of equal consecutive keys\n#\n");
  printf("# \tOptions:\n#\n");
  printf("# \t\t--cpu\t\t Run on CPU serially instead of GPU\n");
  printf("# \t\t--print\t\t Print results of algorithm (check validity)\n");
  printf("# \t\t--thrust\t Run on GPU but using thrust library\n");

  printf("# \t\t--load\t Use 32-bit texture data s\n");
  printf("# \t\t--rnd\t Take uniform random keys s\n");
}


static
unsigned int* MyTexture_load(char* filename, int* dataSize)
{
    FILE* file = fopen(filename, "rb");
    //texture->dataRGBA8888 = NULL;
    if (!file)
    {
        char* tmp = (char*)malloc(strlen(filename) + 10);
        if (tmp)
        {
            char* ptr = tmp;
            strcpy(ptr, "../");
            ptr += 3;
            strcpy(ptr, filename);
            file = fopen(tmp, "rb");
        }
    }
    // Read
    if (file)
    {
        int npixels = 512 * 512;//texture->width * texture->height;
        int size = npixels * 4;
        unsigned int* data = (unsigned int*)malloc(size);
        *dataSize = npixels;
        if (data)
        {
            int i;
            for (i = 0; i < npixels; i++)
            {
                unsigned int r, g, b;
                unsigned int raw = 0;
                unsigned int pixel = 0;
                int rsize = fread(&raw, 3, 1, file);
                if (rsize != 1)
                {
                    printf(
                        "Warning: Unexpected EOF in texture %s at idx %d\n",
                        filename, i);
                    break;
                }
                r = (raw & 0x00FF0000) >> 16;
                g = (raw & 0x0000FF00) >> 8;
                b = (raw & 0x000000FF) >> 0;
                pixel = 0xFF000000 | (b << 16) | (g << 8) | (r << 0);
                data[i] = pixel;
            }
        }
        fclose(file);
        return data;
    }
    return NULL;
}




static inline int getInput(size_t i, unsigned int* texData, int dataSize, bool rnd, int nIndices)
{
  if (texData)
  {
    static size_t index = i % dataSize;
    static unsigned int round = 0;
    unsigned int val = texData[index];
    unsigned int result;

    result = val + round;
    index++;
    if (index >= dataSize)
    {
        index = 0;
        round += 7;
    }
    return result % nIndices;
  }
  else
  {
    static unsigned int current = 0xf1232345;
    const unsigned int mult = 1664525;
    const unsigned int add = 1013904223ul;

    current = current * mult + add;
    if (!rnd)
        current = i / 100;
    i = (int)(current);
    i = i % nIndices;
    return i;
  }
}

static void fillInput(int* input, bool load, bool rnd, int nIndices)
{
  size_t i;
  unsigned int* texData = NULL;
  int dataSize = 0;
  if (load && !rnd)
  {
      texData = MyTexture_load("texture.raw", &dataSize);
  }
  if (nIndices > 256)
  {
      for (i = 0; i < TEST_SIZE * 4;)
      {
        *input++ = getInput(i++, texData, dataSize, rnd, nIndices);
        *input++ = getInput(i++, texData, dataSize, rnd, nIndices);
        *input++ = getInput(i++, texData, dataSize, rnd, nIndices);
        *input++ = getInput(i++, texData, dataSize, rnd, nIndices);
      }
  }
  else
  {
      unsigned char * byteInput = (unsigned char * )input;
      for (i = 0; i < TEST_SIZE * 16;)
      {
        *byteInput++ = getInput(i++, texData, dataSize, rnd, nIndices);
        *byteInput++ = getInput(i++, texData, dataSize, rnd, nIndices);
        *byteInput++ = getInput(i++, texData, dataSize, rnd, nIndices);
        *byteInput++ = getInput(i++, texData, dataSize, rnd, nIndices);
      }

  }
  if (texData) free(texData);
}


static const char* compModeToString(int compMode)
{
  const char* compModeStr;
  switch (compMode)
  {
    case cudaComputeModeDefault:
      compModeStr = "Not restricted";
      break;
    case cudaComputeModeExclusive:
      compModeStr = "Exclusive";
      break;
    case cudaComputeModeProhibited:
      compModeStr = "Not available";
      break;
    default:
      compModeStr = "Unknown mode";
      break;
  }
  return compModeStr;
}

static
void printGPUData(void)
{
    int devId;
    cudaDeviceProp props;

    enum cudaError cudaErr = cudaGetDevice( &devId );

    if(!cudaErr)
        cudaErr = cudaGetDeviceProperties( &props, devId );
    if(cudaErr)
    {
        printf("Error getting device! cudaErr = %s\n", cudaGetErrorString( cudaErr ));
        return;
    }

    // Print important and interesting properties:
    {
        const char* compMode = compModeToString(props.computeMode);
        printf("# Running on device %d: %s\n",devId,props.name);
        printf("# CUDA-Architecture version: %d.%d\t", props.major, props.minor);
        printf("# Tot. global mem  = %lu\t", (unsigned long)props.totalGlobalMem);
        printf("# Tot. const mem  = %lu\n", (unsigned long)props.totalConstMem);
        printf("# Num sms = %d\t", props.multiProcessorCount);
        printf("# Regs / sm (block) = %d\t", props.regsPerBlock);
        printf("# shared mem / sm(block) = %lu\t", (unsigned long)props.sharedMemPerBlock);
        printf("# ClockRate  = %d\n", props.clockRate);
        printf("# ECC Enabled = %d\t", props.ECCEnabled);
        printf("# Overlap kernels = %d\t", props.concurrentKernels);
        printf("# Overlap mem xfer = %d\n", props.deviceOverlap);
        printf("# Computemode = %s\n", compMode);
    }
}


int main (int argc, char** argv)
{
  int i;
  int index_0 = 0;
  int index_1 = index_0 + TEST_SIZE;

  bool cpu = false;
  bool print = false;
  bool thrust = false;

  bool rnd = false;
  bool load = false;

  printUsage();

  for (i = 0; i < argc; i++)
  {
    if (argv[i] && strcmp(argv[i], "--cpu") == 0)
      cpu = true;
    if (argv[i] && strcmp(argv[i], "--print") == 0)
      print = true;
    if (argv[i] && strcmp(argv[i], "--thrust") == 0)
      thrust = true;
    if (argv[i] && strcmp(argv[i], "--load") == 0)
      load = true;
    if (argv[i] && strcmp(argv[i], "--rnd") == 0)
        rnd = true;
  }
  {
    // Allocate keys:
    int* INPUT = NULL;
    int* hostINPUT = (int*)malloc(4 * sizeof(int) * (TEST_SIZE + 3));
    assert(hostINPUT);
    if (!cpu)
    {
      cudaMalloc(&INPUT, 4 * sizeof(int) * TEST_SIZE);
      assert(INPUT);
      // Ok CUDA is up - print out info on chosen GPU:
      printGPUData();
    }
    // Create events for timing:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int test=0; test < sizeof(checkNBins) / sizeof(checkNBins[0]); test++)
    {
        int nIndices = checkNBins[test];
        fillInput(hostINPUT, load, rnd, nIndices);
        if (!cpu)
          cudaMemcpy(INPUT, hostINPUT, 4 * sizeof(int) * TEST_SIZE, cudaMemcpyHostToDevice);
        void* tmpBuffer = NULL;
        int zero = 0;
        int tmpbufsize = getHistogramBufSize<histogram_atomic_inc>(zero , nIndices);
        cudaMalloc(&tmpBuffer, tmpbufsize);
        // Now start timer - we run on stream 0 (default stream):
        cudaEventRecord(start, 0);
        if (thrust)
        {
          #if ENABLE_THRUST
            int* tmpin;
            cudaMalloc(&tmpin, 4 * sizeof(int) * TEST_SIZE);
            for (int run = 0; run < NRUNS; run++){
              cudaMemcpy(tmpin, INPUT, 4 * sizeof(int) * TEST_SIZE, cudaMemcpyDeviceToDevice);
              if (nIndices <= 256)
               testHistogramParamThrust((unsigned char*)tmpin, 16*index_0, 16*index_1, print, nIndices);
              else
               testHistogramParamThrustInt(tmpin, 4*index_0, 4*index_1, print, nIndices);
            }
          #else
            printf("\nTest was compiled without thrust support! Find 'ENABLE_THRUST' in source-code!\n\n Exiting...\n");
            break;
          #endif
        }
        else
        {
          testHistogramParam((uint4*)INPUT, (uint4*)hostINPUT, index_0, index_1, print, cpu, tmpBuffer, nIndices);
        }
        {
            float t_ms;
            cudaEventRecord(stop, 0);
            cudaThreadSynchronize();
            cudaEventElapsedTime(&t_ms, start, stop);
            double t = t_ms * 0.001f;
            int nKeys = TEST_SIZE*4;
            if (nIndices <= 256) nKeys *= 4;
            double GKeys = ((double)nKeys)*1.e-9;
            double GKps = ((double)NRUNS) * GKeys / (double)t;
            double GBps = nIndices > 256 ? GKps * 4.0 : GKps;
            if (test == 0){
                printf("# Nbins <= 256: N keys/run = %.2f MKeys, runs = %d, tot keys = %.2f MKeys\n",
                        GKeys*1000.0, NRUNS, GKeys*((double)NRUNS)*1000.0);
                printf("# Nbins > 256: N keys/run = %.2f MKeys, runs = %d, tot keys = %.2f MKeys\n",
                        GKeys*250.0, NRUNS, GKeys*((double)NRUNS)*250.0);
                printf("# Number of bins, Runtime in loops: (s), Throughput (Gkeys/s), Throughput (GB/s)\n");
            }
            printf("%d,\t\t%4f,\t\t%4f,\t\t%4f,\n", nIndices, t, GKps, GBps);
        }
        if (tmpBuffer) cudaFree(tmpBuffer);
    }
    if (INPUT) cudaFree(INPUT);
    if (hostINPUT) free(hostINPUT);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  return 0;
}

