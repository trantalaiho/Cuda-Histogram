/*
 *
 *
 *  Created on: 27.6.2011
 *      Author: Teemu Rantalaiho (teemu.rantalaiho@helsinki.fi)
 *
 *
 *  Copyright 2011 Teemu Rantalaiho
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
 *   nvcc -O4 -arch=<your_arch> -I../ test_mem_multi16.cu -o test_mem_multi16
 * 
 *
 */

#define TESTMAXIDX   256        // default 256 keys (change this) / indices
#define TEST_IS_POW2 1          // Note: This tells is TESTMAXIDX is power of two number
#define TEST_SIZE (625 * 100 * 1000 )   // 62.5 million inputs -> 1000 million keys
#define NRUNS 100               // Repeat 100 times => 100 Gigainputs in total (16 keys per entry)
#define START_INDEX	0
#define NSTRESS_RUNS    NRUNS

#define ENABLE_THRUST   0   // Enable thrust-based version also (xform-sort_by_key-reduce_by_key)


#include "cuda_histogram.h"

#if ENABLE_THRUST
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#endif

#include <assert.h>
#include <stdio.h>



// Always return 1 -> normal histogram - each sample has same weight
struct test_xform2
{
  __host__ __device__
  void operator() (uint4* input, int i, int* result_index, int* results, int nresults) const {
    uint4 idata = input[i];
    #pragma unroll
    for (int resIdx = 0; resIdx < 4; resIdx++)
    {
        unsigned int data = ((unsigned int*)(&idata))[resIdx];
    #if TESTMAXIDX < 256
    #if TEST_IS_POW2
        *result_index++ = ((data >> 24) & 0xFF) & (TESTMAXIDX - 1);
        *result_index++ = ((data >> 16) & 0xFF) & (TESTMAXIDX - 1);
        *result_index++ = ((data >>  8) & 0xFF) & (TESTMAXIDX - 1);
        *result_index++ = ((data >>  0) & 0xFF) & (TESTMAXIDX - 1);
    #else
        *result_index++ = ((data >> 24) & 0xFF) % (TESTMAXIDX);
        *result_index++ = ((data >> 16) & 0xFF) % (TESTMAXIDX);
        *result_index++ = ((data >>  8) & 0xFF) % (TESTMAXIDX);
        *result_index++ = ((data >>  0) & 0xFF) % (TESTMAXIDX);
    #endif
    #else
        *result_index++ = ((data >> 24));
        *result_index++ = ((data >> 16) & 0xFF);
        *result_index++ = ((data >>  8) & 0xFF);
        *result_index++ = ((data >>  0) & 0xFF);
    #endif
        *results++ = 1;
        *results++ = 1;
        *results++ = 1;
        *results++ = 1;
    }
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

static void testHistogramParam(uint4* INPUT, uint4* hostINPUT, int index_0, int index_1, bool print, bool cpurun, bool stress, void* tmpBuffer)
{
  int nIndex = TESTMAXIDX;
  int srun;
  int nruns = stress ? NSTRESS_RUNS : 1;
  test_sumfun2 sumFun;
  test_xform2 transformFun;
  //test_indexfun2 indexFun;
  int* tmpres = (int*)malloc(sizeof(int) * nIndex);
  int* cpures = stress ? (int*)malloc(sizeof(int) * nIndex) : tmpres;
  int zero = 0;
  for (srun = 0; srun < nruns; srun++)
  {
    {
      int* tmpidx = (int*)malloc(sizeof(int) * nIndex);
      if (print)
        printf("\nTest reduce_by_key:\n\n");
      memset(tmpres, 0, sizeof(int) * nIndex);
      if (stress)
          memset(cpures, 0, sizeof(int) * nIndex);
      if (cpurun || stress)
        for (int i = index_0; i < index_1; i++)
        {
          int index[16];
          int tmp[16];
          transformFun(hostINPUT, i, &index[0], &tmp[0], 1);
          //index = indexFun(INPUT, i);
          for (int tmpi = 0; tmpi < 16; tmpi++)
              cpures[index[tmpi]] = sumFun(cpures[index[tmpi]], tmp[tmpi]);
          //printf("i = %d,  out_index = %d,  out_val = (%.3f, %.3f) \n",i, index, tmp.real, tmp.imag);
        }
      if (print && cpurun)
      {
          printres(cpures, nIndex, "CPU results:");
      }
    }

    if (!cpurun)
      callHistogramKernel<histogram_atomic_inc, 16>(INPUT, transformFun, /*indexFun,*/ sumFun, index_0, index_1, zero, tmpres, nIndex, false, 0, tmpBuffer);
    if (stress)
    {
        int k;
        for (k = 0; k < nIndex; k++)
        {
            if (tmpres[k] != cpures[k] /*|| tmpres[k].imag != cpures[k].imag*/)
            {
                printf("Error detected with index-values: i0 = %d, i1 = %d!\n", index_0, index_1);
                printres(cpures, nIndex, "CPU results:");
                printres(tmpres, nIndex, "GPU results:");
            }
        }
    }

    if (print && (!cpurun))
    {
      printres(tmpres, nIndex, "GPU results:");
    }
    int size  = index_1 - index_0;
    index_0 += 1;
    index_1 -= 1;
    if (index_0 > index_1 + 1)
    {
        int tmp = index_0;
        index_0 = index_1;
        index_1 = tmp;
    }
    if (index_0 < 0 || index_1 < 0) {
      index_0 = 0;
      index_1 = size - 1;
    }
  }
  free(tmpres);
  if (stress)
    free(cpures);
}

#if ENABLE_THRUST

// NOTE: Take advantage here of the fact that this is the classical histogram with all values = 1
// And also that we know before hand the number of indices coming out
static void testHistogramParamThrust(int* INPUT, int index_0, int index_1, bool print)
{
  test_sumfun2 mysumfun;
  thrust::equal_to<int> binary_pred;
  int nIndex = TESTMAXIDX;
  int N = index_1 - index_0;
  thrust::device_vector<int> keys_out(nIndex);
  thrust::device_vector<int> vals_out(nIndex);
  thrust::device_vector<int> h_vals_out(nIndex);
  //thrust::device_vector<int> keys(N);
  thrust::device_ptr<int> keys(INPUT);
  // Sort the data
  thrust::sort(keys, keys + N);
  // And reduce by key - histogram complete
  thrust::reduce_by_key(keys, keys + N, thrust::make_constant_iterator(1), keys_out.begin(), vals_out.begin(), binary_pred, mysumfun);
  h_vals_out = vals_out;
  if (print)
  {
    printf("\nThrust results:\n");
    printf("vals = [ ");
    for (int i = 0; i < nIndex; i++)
    {
        int tmp = h_vals_out[i];
        printf("(%d), ", tmp);
    }
    printf("]\n");
  }
}
#endif

void printUsage(void)
{
  printf("\n");
  printf("Test order independent reduce-by-key / histogram algorithm\n\n");
  printf("By default this runs on custom algorithm on the GPU, with lots of equal consecutive keys\n\n");
  printf("\tOptions:\n\n");
  printf("\t\t--cpu\t\t Run on CPU serially instead of GPU\n");
  printf("\t\t--print\t\t Print results of algorithm (check validity)\n");
  printf("\t\t--thrust\t Run on GPU but using thrust library\n");

  printf("\t\t--load\t Use 32-bit texture data s\n");
  printf("\t\t--rnd\t Take uniform random keys s\n");
//  printf("\t\t--sharp\t Make peaks sharp\n");
//  printf("\t\t--nornd\t Remove random noise from input\n");
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




static inline int getInput(size_t i, unsigned int* texData, int dataSize, bool rnd)
{
  if (texData)
  {
    static size_t index = i % dataSize;
    static int round = 0;
    unsigned int val = texData[index];
    int result;

    result = val + round;
    index++;
    if (index >= dataSize)
    {
        index = 0;
        round += 7;
    }

#if TEST_IS_POW2
    result = (int)(result);
#else
    result = (int)(result);
#endif
    return result;
  }
  else
  {
    static unsigned int current = 0xf1232345;
    const unsigned int mult = 1664525;
    const unsigned int add = 1013904223ul;

    //int peakWidth = sharp ? 100 : 10000;
//    static int nextPeak = 200;
//    static int currentBase = 0;
//    static int currentWidth = TESTMAXIDX;

    current = current * mult + add;
    if (!rnd)
        current = i / 100;
#if TEST_IS_POW2
    i = (int)(current);
#else
    i = (int)(current);
#endif
    return i;
  }
}

static void fillInput(int* input, bool load, bool rnd)
{
  size_t i;
  unsigned int* texData = NULL;
  int dataSize = 0;
  if (load && !rnd)
  {
      texData = MyTexture_load("texture.raw", &dataSize);
  }
  for (i = 0; i < TEST_SIZE * 4;)
  {
    *input++ = getInput(i++, texData, dataSize, rnd);
    *input++ = getInput(i++, texData, dataSize, rnd);
    *input++ = getInput(i++, texData, dataSize, rnd);
    *input++ = getInput(i++, texData, dataSize, rnd);
  }
  if (texData) free(texData);
}



int main (int argc, char** argv)
{
  int i;
  int index_0 = START_INDEX;
  int index_1 = index_0 + TEST_SIZE;

  bool cpu = false;
  bool print = false;
  bool thrust = false;
  bool stress = false;

//  bool peaks = false;
//  bool sharp = false;
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
    if (argv[i] && strcmp(argv[i], "--stress") == 0)
      stress = true;
//    if (argv[i] && strcmp(argv[i], "--peaks") == 0)
 //     peaks = true;
    if (argv[i] && strcmp(argv[i], "--load") == 0)
      load = true;
 //   if (argv[i] && strcmp(argv[i], "--sharp") == 0)
 //       sharp = true;
    if (argv[i] && strcmp(argv[i], "--rnd") == 0)
        rnd = true;
  }
  {
    // Allocate keys:
    int* INPUT = NULL;
    int* hostINPUT = (int*)malloc(4 * sizeof(int) * (TEST_SIZE + 3));
    assert(hostINPUT);
    fillInput(hostINPUT, load, rnd);
    if (!cpu)
    {
      cudaMalloc(&INPUT, 4 * sizeof(int) * TEST_SIZE);
      assert(INPUT);
      cudaMemcpy(INPUT, hostINPUT, 4 * sizeof(int) * TEST_SIZE, cudaMemcpyHostToDevice);
    }
    void* tmpBuffer = NULL;
    int zero = 0;
    int tmpbufsize = getHistogramBufSize<histogram_atomic_inc>(zero , (int)TESTMAXIDX);
    cudaMalloc(&tmpBuffer, tmpbufsize);

    // Create events for timing:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Now start timer - we run on stream 0 (default stream):
    cudaEventRecord(start, 0);

    for (i = 0; i < NRUNS; i++)
    {
      if (thrust)
      {
        #if ENABLE_THRUST
          testHistogramParamThrust(INPUT, index_0, index_1, print);
        #else
          printf("\nTest was compiled without thrust support! Find 'ENABLE_THRUST' in source-code!\n\n Exiting...\n");
          break;
        #endif
      }
      else
      {
        testHistogramParam((uint4*)INPUT, (uint4*)hostINPUT, index_0, index_1, print, cpu, stress, tmpBuffer);
      }
      print = false;
      // Run only once all stress-tests
      if (stress) break;
    }
    {
        float t_ms;
        cudaEventRecord(stop, 0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&t_ms, start, stop);
        double t = t_ms * 0.001f;
        double GKps = (((double)TEST_SIZE * (double)NRUNS * 16.0)) / (t*1.e9);
        printf("Runtime in loops: %fs, Throughput (Gkeys/s): %3f GK/s \n", t, GKps);
    }
    if (INPUT) cudaFree(INPUT);
    if (hostINPUT) free(hostINPUT);
    if (tmpBuffer) cudaFree(tmpBuffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  return 0;
}

