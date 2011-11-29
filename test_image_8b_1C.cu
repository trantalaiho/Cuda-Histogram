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
 *
 *
 */

#define TESTMAXIDX   256      // 256 keys / indices
#define TEST_IS_POW2 1
#define NRUNS 1000

#define ENABLE_THRUST   0   // Enable thrust-based version also (xform-sort_by_key-reduce_by_key)
#ifndef NONPP
#define ENABLE_NPP      1   // NOTE: In order to link, use: -lnpp -L /usr/local/cuda/lib64/ (or similar)
#endif

#ifdef OLDARCH
#define FOR_PRE_FERMI   1
#else
#define FOR_PRE_FERMI   0
#endif


#if ENABLE_NPP
#include <npp.h>

/*#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>
*/
#include <string>
#include <assert.h>
#include <iostream>
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
#endif


#include <stdio.h>

static bool csv = false;

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
#if 0
        int r = (data >> 16) & 0xFF;
        int g = (data >>  8) & 0xFF;
        int b = (data >>  0) & 0xFF;
        const float scale = (float)(TESTMAXIDX - 1) / (float)(255 + 255 + 255);
        float x = (float)(r + g + b);
        float result = x * scale;
        *result_index++ = (int)result;
#else
        *result_index++ = (data >> 24);
        *result_index++ = (data >> 16) & 0XFF;
        *result_index++ = (data >>  8) & 0XFF;
        *result_index++ = (data >>  0) & 0XFF;
#endif
        *results++ = 1;
        *results++ = 1;
        *results++ = 1;
        *results++ = 1;
    }
  }
};

// Always return 1 -> normal histogram - each sample has same weight
struct test_xform_old
{
  __host__ __device__
  void operator() (unsigned int* input, int i, int* result_index, int* results, int nresults) const {
    unsigned int data = input[i];
    *result_index++ = (data >> 24);
    *result_index++ = (data >> 16) & 0XFF;
    *result_index++ = (data >>  8) & 0XFF;
    *result_index++ = (data >>  0) & 0XFF;
    *results++ = 1;
    *results++ = 1;
    *results++ = 1;
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
    if (csv)
    {
      printf("vals = [\n");
      for (int i = 0; i < nres; i++)
          printf("%d\n", res[i]);
      printf("]\n");
    }
    else
    {
      printf("vals = [ ");
      for (int i = 0; i < nres; i++)
          printf(" %d, ", res[i]);
      printf("]\n");
    }
}

static void testHistogram(uint4* INPUT, uint4* hostINPUT, int nPixels,  bool print, bool cpurun, bool npp, void* nppSize, void* nppBuffer, void* nppResBuffer)
{
  int nIndex = TESTMAXIDX;
  test_sumfun2 sumFun;
  test_xform2 transformFun;
  test_xform_old transformFunOld;
  //test_indexfun2 indexFun;
  int* tmpres = (int*)malloc(sizeof(int) * nIndex);
  int* cpures = tmpres;
  int zero = 0;
  {
    {
      if (print)
        printf("\nTest reduce_by_key:\n\n");
      memset(tmpres, 0, sizeof(int) * nIndex);
      if (cpurun)
        for (int i = 0; i < (nPixels >> 2); i++)
        {
          int index[16];
          int tmp[16];
          transformFun(hostINPUT, i, &index[0], &tmp[0], 16);
          for (int tmpi = 0; tmpi < 16; tmpi++)
              cpures[index[tmpi]] = sumFun(cpures[index[tmpi]], tmp[tmpi]);
          //printf("i = %d,  out_index = %d,  out_val = (%.3f, %.3f) \n",i, index, tmp.real, tmp.imag);
        }
      if (print && cpurun)
      {
          printres(cpures, nIndex, "CPU results:");
      }
    }

    if (!cpurun && !npp)
    {
#if FOR_PRE_FERMI
      callHistogramKernel<histogram_atomic_inc, 4>((unsigned int*)INPUT, transformFunOld, /*indexFun,*/ sumFun, 0, (nPixels), zero, (int*)nppResBuffer, nIndex, true, 0, nppBuffer);
#else
      callHistogramKernel<histogram_atomic_inc, 16>(INPUT, transformFun, /*indexFun,*/ sumFun, 0, (nPixels >> 2), zero, (int*)nppResBuffer, nIndex, true, 0, nppBuffer);
#endif
      cudaMemcpy(tmpres, nppResBuffer, sizeof(int) * TESTMAXIDX, cudaMemcpyDeviceToHost);
    }
    else if (npp)
    {
#if ENABLE_NPP
        NppiSize oSizeROI = *(NppiSize*)nppSize;
        Npp8u* pDeviceBuffer = (Npp8u*)nppBuffer;
        nppiHistogramEven_8u_C1R(
            (Npp8u*)INPUT, oSizeROI.width, oSizeROI,
            (Npp32s*)nppResBuffer, TESTMAXIDX + 1, 0, TESTMAXIDX,
            pDeviceBuffer);
        cudaMemcpy(tmpres, nppResBuffer, sizeof(int) * TESTMAXIDX, cudaMemcpyDeviceToHost);
#endif
    }

    if (print && (!cpurun))
    {
      printres(tmpres, nIndex, "GPU results:");
    }

  }
  free(tmpres);
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
  printf("Test order independent reduce-by-key / histogram algorithm with an image histogram\n\n");
  printf("\tOptions:\n\n");
  printf("\t\t--cpu\t\t Run on CPU serially instead of GPU\n");
  printf("\t\t--print\t\t Print results of algorithm (check validity)\n");
  printf("\t\t--thrust\t Run on GPU but using thrust library\n");
  printf("\t\t--csv\t\t When printing add line-feeds to ease openoffice import...\n");
  printf("\t\t--npp\t\t Use NVIDIA Performance Primitives library (NPP) instead.\n");

  printf("\t\t--load <name>\t Use 32-bit texture data s\n");
}





static void fillInput(int* input, const char* filename, int nPixels)
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
      unsigned int* data = (unsigned int*)input;
      if (data)
      {
          int i;
          for (i = 0; i < nPixels; i++)
          {
              unsigned int raw = 0;
              int rsize = fread(&raw, 3, 1, file);
              if (rsize != 1)
              {
                  printf(
                      "Warning: Unexpected EOF in texture %s at idx %d\n",
                      filename, i);
                  break;
              }
              data[i] = (raw & 0x00FFFFFF) | ((i & 0xFF) << 24);
/*              r = (raw & 0x00FF0000) >> 16;
              g = (raw & 0x0000FF00) >> 8;
              b = (raw & 0x000000FF) >> 0;
              pixel = 0xFF000000 | (b << 16) | (g << 8) | (r << 0);
              data[i] = pixel;*/
          }
      }
      fclose(file);
  }
}

#include <sys/time.h>
#include <sys/stat.h>


static double cputime_fast()
{
  struct timeval resource;
  gettimeofday(&resource,NULL);
  return( resource.tv_sec + 1.0e-6*resource.tv_usec );
}


int main (int argc, char** argv)
{
  int i;

  bool cpu = false;
  bool print = false;
  bool thrust = false;
  bool npp = false;

  const char* name = "feli.raw";

  double t0;

  printUsage();

  for (i = 0; i < argc; i++)
  {
    if (argv[i] && strcmp(argv[i], "--cpu") == 0)
      cpu = true;
    else if (argv[i] && strcmp(argv[i], "--csv") == 0)
      csv = true;
    else if (argv[i] && strcmp(argv[i], "--npp") == 0)
      npp = true;
    else if (argv[i] && strcmp(argv[i], "--print") == 0)
      print = true;
    else if (argv[i] && strcmp(argv[i], "--thrust") == 0)
      thrust = true;
    else if (argv[i] && strcmp(argv[i], "--load") == 0)
    {
      if (argc > i + 1)
        name = argv[i + 1];
    }
  }

  {

    int nPixels = 0;
    {
      struct stat fileStat;
      int error = stat(name, &fileStat);
      if (error == 0)
      {
        nPixels = (int)((fileStat.st_size / 12) << 2);
        if (nPixels <= 0)
        {
          printf("Filesize is too large or small...Sorry...\n");
          return 1;
        }
      }
      else
      {
        printf("Can't access file: %s, errorcode = %d (man stat)\n", name, error);
        return error;
      }
    }

    // Allocate keys:
    int* INPUT = NULL;
    int* hostINPUT = (int*)malloc(sizeof(int) * nPixels);
    void* nppBuffer = NULL;
    void* nppResBuffer = NULL;
    void* nppSize = NULL;
#if (ENABLE_NPP == 1)
    NppiSize oSizeROI = {0, 0};
    nppSize = &oSizeROI;
#endif
    assert(hostINPUT);
    fillInput(hostINPUT, name, nPixels);
    if (!cpu)
    {
      cudaMalloc(&INPUT, sizeof(int) * nPixels);
      assert(INPUT);
      cudaMemcpy(INPUT, hostINPUT, sizeof(int) * nPixels, cudaMemcpyHostToDevice);
      cudaMalloc(&nppResBuffer, sizeof(int) * TESTMAXIDX);
      cudaMemset(nppResBuffer, 0, sizeof(int) * TESTMAXIDX);
    }
    if (npp)
    {
      #if (ENABLE_NPP == 0)
            printf("Sorry - you did not compile with npp-support -bailing out...\n");
            return 2;
      #else
            int nDeviceBufferSize;
            int levelCount = TESTMAXIDX + 1;
            // Start guessing from 4096 and div by two
            int width = 4096;
            int height = nPixels / width;
            while (width > 128)
            {
                if (width * height == nPixels)
                    break;
                width >>= 1;
                height = nPixels / width;
            }
            oSizeROI.width = width*4;
            oSizeROI.height = height;
            nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, levelCount ,&nDeviceBufferSize);
            cudaMalloc(&nppBuffer, nDeviceBufferSize);
      #endif
    }
    else
    {
        int zero = 0;
        int tmpbufsize = getHistogramBufSize<histogram_atomic_inc>(zero , (int)TESTMAXIDX);
        cudaMalloc(&nppBuffer, tmpbufsize);
    }

    // Now start timer:
    t0 = cputime_fast();

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
        testHistogram((uint4*)INPUT, (uint4*)hostINPUT, nPixels, print, cpu, npp, nppSize, nppBuffer, nppResBuffer);
      }
      print = false;
      // Run only once all stress-tests
    }
    {
        double t = cputime_fast() - t0;
        printf("Runtime in loops: %fs\n", t);
    }
    if (INPUT) cudaFree(INPUT);
    if (hostINPUT) free(hostINPUT);
    if (nppBuffer) cudaFree(nppBuffer);
  }
  return 0;
}

