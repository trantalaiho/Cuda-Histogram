/*
 * test.cu
 *
 *  Created on: 8.3.2012
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
 *
 *
 */

/*
 *
 * Compile with:
 *
 *  nvcc -O4 -arch=<your_arch> -I../ test_indices.cu -o test_indices
 *
 *  Optionally include -DTHRUST to enable thrust codepath
 */


#define TESTMAXIDX      32            // 32 keys / indices
#define NRUNS           40           // Repeat 40 times
#define TEST_SIZE       (20 * 1000 * 1000)   // 20 million inputs
#define START_INDEX	    (1ull << 33) // Big number
#define NSTRESS_RUNS    NRUNS

#ifdef THRUST
#define ENABLE_THRUST   1   // Enable thrust-based version also (xform-sort_by_key-reduce_by_key)
#else
#define ENABLE_THRUST   0   // Enable thrust-based version also (xform-sort_by_key-reduce_by_key)
#endif

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


typedef struct myTestType_s
{
  unsigned long long real;
  int imag;
} myTestType;


struct test_xform2
{
  __host__ __device__
  void operator() (int input, unsigned long long i, int* result_index, myTestType* results, int nRes) const {
    int tmp;
    results->real = i;
    results->imag = input;
    tmp = 10*i*i/(100+i) - i + input;
    if (tmp < 0)
      tmp = -tmp;
    *result_index = tmp % TESTMAXIDX;
  }
};


struct test_sumfun2 {
  __device__ __host__
  myTestType operator() (myTestType RESULT_NAME, myTestType TMP_RESULT) const{
    RESULT_NAME.real += TMP_RESULT.real;
    RESULT_NAME.imag += TMP_RESULT.imag;
    return RESULT_NAME;
  }
};


static void printres (myTestType* res, int nres, const char* descr)
{
    if (descr)
        printf("\n%s:\n", descr);
    printf("vals = [ ");
    for (int i = 0; i < nres; i++)
        printf("(%llu, %d), ", res[i].real, res[i].imag);
    printf("]\n");
}

static void testHistogramParam(int INPUT, unsigned long long index_0, unsigned long long index_1, bool print, bool cpurun, bool stress)
{
  int nIndex = TESTMAXIDX;
  int srun;
  int nruns = stress ? NSTRESS_RUNS : 1;
  test_sumfun2 sumFun;
  test_xform2 transformFun;
  //test_indexfun2 indexFun;
  myTestType* tmpres = (myTestType*)malloc(sizeof(myTestType) * nIndex);
  myTestType* cpures = stress ? (myTestType*)malloc(sizeof(myTestType) * nIndex) : tmpres;
  myTestType zero = {0};
  for (srun = 0; srun < nruns; srun++)
  {
    {
      int* tmpidx = (int*)malloc(sizeof(int) * nIndex);
      if (print)
        printf("\nTest reduce_by_key:\n\n");
      memset(tmpres, 0, sizeof(myTestType) * nIndex);
      if (stress)
          memset(cpures, 0, sizeof(myTestType) * nIndex);
      if (cpurun || stress)
        for (unsigned long long i = index_0; i < index_1; i++)
        {
          int index;
          myTestType tmp;
          transformFun(INPUT, i, &index, &tmp, 1);
          //index = indexFun(INPUT, i);
          cpures[index] = sumFun(cpures[index], tmp);
          //printf("i = %d,  out_index = %d,  out_val = (%.3f, %.3f) \n",i, index, tmp.real, tmp.imag);
        }
      if (print && cpurun)
      {
          printres(cpures, nIndex, "CPU results:");
      }
    }

    if (!cpurun)
      callHistogramKernel<histogram_generic, 1>
        (INPUT, transformFun, sumFun, index_0, index_1, zero, tmpres, nIndex);

    if (stress)
    {
        int k;
        for (k = 0; k < nIndex; k++)
        {
            if (tmpres[k].real != cpures[k].real || tmpres[k].imag != cpures[k].imag)
            {
                printf("Error detected with index-values: i0 = %d, i1 = %d!\n", index_0, index_1);
                printres(cpures, nIndex, "CPU results:");
                printres(tmpres, nIndex, "GPU results:");
                break;
            }
        }
    }

    if (print && (!cpurun))
    {
      printres(tmpres, nIndex, "GPU results:");
    }
    int size  = index_1 - index_0;
    index_0 += size;
    index_1 += size + 1;
  }
  free(tmpres);
  if (stress)
    free(cpures);
}

#if ENABLE_THRUST
struct initFunObjType
{
  __device__ __host__
  thrust::tuple<int, myTestType> operator() (thrust::tuple<int, unsigned long long> t) const {
    int INPUT = thrust::get<0>(t);
    unsigned long long index = thrust::get<1>(t);
    test_xform2 xformFun;
    int out_index;
    myTestType res;
    xformFun(INPUT, index, &out_index, &res, 1);
    return thrust::make_tuple(out_index, res);
  }
};

static
void initialize(thrust::device_vector<int>& keys, thrust::device_vector<myTestType>& vals, unsigned long long index_0, unsigned long long index_1, int INPUT)
{
thrust::counting_iterator<unsigned long long, thrust::device_space_tag> idxBegin(index_0);
thrust::counting_iterator<unsigned long long, thrust::device_space_tag> idxEnd(index_1);
initFunObjType initFun;
    thrust::transform(
        thrust::make_zip_iterator(make_tuple(
            thrust::make_constant_iterator(INPUT) + index_0,
            idxBegin)),
        thrust::make_zip_iterator(make_tuple(
            thrust::make_constant_iterator(INPUT) + index_1,
            idxEnd)),
        thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), vals.begin())),
        initFun);

}

static void testHistogramParamThrust(int INPUT, unsigned long long index_0, unsigned long long index_1, bool print)
{
  test_sumfun2 mysumfun;
  thrust::equal_to<int> binary_pred;
  int nIndex = TESTMAXIDX;
  unsigned long long N = index_1 - index_0;
  thrust::device_vector<int> keys_out(nIndex);
  thrust::device_vector<myTestType> vals_out(nIndex);
  thrust::host_vector<myTestType> h_vals_out(nIndex);
  thrust::device_vector<int> keys(N);
  thrust::device_vector<myTestType> values(N);
  initialize(keys, values, index_0, index_1, INPUT);
  thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
  thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_out.begin(), vals_out.begin(), binary_pred, mysumfun);
  h_vals_out = vals_out;
  if (print)
  {
    printres(&h_vals_out[0], nIndex, "Thrust results:");
  }
}
#endif

void printUsage(void)
{
  printf("\n");
  printf("Test order independent reduce-by-key / histogram algorithm\n\n");
  printf("By default this runs on custom algorithm on the GPU\n\n");
  printf("\tOptions:\n\n");
  printf("\t\t--cpu\t\t Run on CPU serially instead of GPU\n");
  printf("\t\t--print\t\t Print results of algorithm (check validity)\n");
  printf("\t\t--thrust\t Run on GPU but using thrust library\n");
}

int main (int argc, char** argv)
{
  int i;
  unsigned long long index_0 = START_INDEX;
  unsigned long long index_1 = index_0 + TEST_SIZE;
  int INPUT = 1;

  bool cpu = false;
  bool print = false;
  bool thrust = false;
  bool stress = false;

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
  }
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
      testHistogramParam(INPUT, index_0, index_1, print, cpu, stress);
    }
    print = false;
    // Run only once all stress-tests
    if (stress) break;
  }
}
