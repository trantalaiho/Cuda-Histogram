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
 *    Compile with:
 *
 *     nvcc -O4 -arch=<your_arch> -I../ test_stress_multi.cu -o test_stress_multi
 * 
 *
 */

#define TESTMAXIDX   6855      // 16 keys / indices
#define TEST_IS_POW2 0
#define TEST_SIZE (1)   // 1000 million inputs
#define NRUNS 334          // Repeat 100 times => 100 gigainputs
#define START_INDEX	2
#define NSTRESS_RUNS    NRUNS
#define START_NBINS	1
#define ENABLE_THRUST   0   // Enable thrust-based version also (xform-sort_by_key-reduce_by_key)
#define NBIN_INC	79
#include <assert.h>
#include <stdio.h>
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



typedef struct myTestType_s
{
  int real;
  int imag;
} myTestType;

#define TEST_JENKINS_MIX(A, B, C)  \
do {                                \
  A -= B; A -= C; A ^= (C>>13);     \
  B -= C; B -= A; B ^= (A<<8);      \
  C -= A; C -= B; C ^= (B>>13);     \
  A -= B; A -= C; A ^= (C>>12);     \
  B -= C; B -= A; B ^= (A<<16);     \
  C -= A; C -= B; C ^= (B>>5);      \
  A -= B; A -= C; A ^= (C>>3);      \
  B -= C; B -= A; B ^= (A<<10);     \
  C -= A; C -= B; C ^= (B>>15);     \
} while (0)

struct test_xform2
{
  __host__ __device__
  void operator() (int input, size_t i, int* result_index, myTestType* result, int nRes) const {
    //int tmp = 1013904223;
    result[0].real = 1;//(i & 0x1) != 0 ? 1 : i;
    result[0].imag = i - 10000;//(i & 0x1) != 0 ? 1 : i;
    result[1].real = i - input;//(i & 0x1) != 0 ? 1 : i;
    result[1].imag = 2;//(i & 0x1) != 0 ? 1 : i;
    result[2].real = (result[0].real + result[1].real) >> 1;//(i & 0x1) != 0 ? 1 : i;
    result[2].imag = (result[0].imag + result[1].imag) >> 1;//(i & 0x1) != 0 ? 1 : i;
    // NOTE: Use i directly instead of x to reveal a nasty compiler bug in nvcc v. 3.2.16 with -arch=sm_20
    int x = i;
    if (0 && (i & 15 == 0))
    {
      int a = 1013904223;
      int b = 1013904223;
      TEST_JENKINS_MIX(a,b,x);
    }
    else if ((i & 0x7) != 0)
    {
      x = x >> 10;
    }
    if (x < 0) x = -x;
    result_index[0] = x % input;
    if (0 && (x & 15 == 0))
    {
      int a = 1013904223;
      int b = 1013904223;
      TEST_JENKINS_MIX(a,b,x);
    }
    else if ((x & 0x7) != 0)
    {
      x = x >> 10;
    }
    if (x < 0) x = -x;
    result_index[1] =  x % input;

    if (0 && (x & 15 == 0))
    {
      int a = 1013904223;
      int b = 1013904223;
      TEST_JENKINS_MIX(a,b,x);
    }
    else if ((x & 0x7) != 0)
    {
      x = x >> 10;
    }
    if (x < 0) x = -x;
    result_index[2] =  x % input;

/*    tmp = i + input;
    if (tmp < 0)
      tmp = -tmp;
    // divide by 16
    //tmp >>= 4;
#if !TEST_IS_POW2
    *result_index = tmp % TESTMAXIDX;
#else
    *result_index = tmp & (TESTMAXIDX-1);
#endif*/
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
        printf("(%d, %d), ", res[i].real, res[i].imag);
    printf("]\n");
}

static bool testHistogramParam(int INPUT, int index_0, int index_1, bool print, bool cpurun, bool stress)
{
  int nIndex = INPUT;
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
      if (print)
        printf("\nTest reduce_by_key:\n\n");
      memset(tmpres, 0, sizeof(myTestType) * nIndex);
      if (stress)
          memset(cpures, 0, sizeof(myTestType) * nIndex);
      if (cpurun || stress)
        for (int i = index_0; i < index_1; i++)
        {
          int index[3];
          myTestType tmp[3];
          transformFun(INPUT, i, &index[0], &tmp[0], 3);
          //index = indexFun(INPUT, i);
          for (int j = 0; j < 3; j++)
            cpures[index[j]] = sumFun(cpures[index[j]], tmp[j]);
          //printf("i = %d,  out_index = %d,  out_val = (%.3f, %.3f) \n",i, index, tmp.real, tmp.imag);
        }
      if (print && cpurun)
      {
          printres(cpures, nIndex, "CPU results:");
      }
    }

    if (stress) printf(" %d ", index_1 - index_0);
    if (!cpurun)
      callHistogramKernel<histogram_generic, 3>(INPUT, transformFun, /*indexFun,*/ sumFun, index_0, index_1, zero, tmpres, nIndex);

    if (stress)
    {
        int k;
        for (k = 0; k < nIndex; k++)
        {
            if (tmpres[k].real != cpures[k].real || tmpres[k].imag != cpures[k].imag)
            {
                printf("Error detected with index-values: i0 = %d, i1 = %d, nbins = %d!\n", index_0, index_1, nIndex);
                printres(cpures, nIndex, "CPU results:");
                printres(tmpres, nIndex, "GPU results:");
                return false;
            }
        }
    }

    if (print && (!cpurun))
    {
      printres(tmpres, nIndex, "GPU results:");
    }
    int size  = index_1 - index_0;
    //index_0 += size;
    index_1 += 1;
    if (srun % 23 == 19)
        index_1 +=  (4 * size / 3) + (size % 73);
    if (index_0 < 0 || index_1 < 0) {
      index_0 = 0;
      index_1 = size - 1;
    }
  }
  free(tmpres);
  if (stress)
    free(cpures);
  return true;
}

#if ENABLE_THRUST
struct initFunObjType
{
  __device__ __host__
  thrust::tuple<int, myTestType> operator() (thrust::tuple<int, int> t) const {
    int INPUT = thrust::get<0>(t);
    int index = thrust::get<1>(t);
    test_xform2 xformFun;
    int out_index;
    myTestType res = xformFun(INPUT, index, &out_index);
    return thrust::make_tuple(out_index, res);
  }
};

static
void initialize(thrust::device_vector<int>& keys, thrust::device_vector<myTestType>& vals, int index_0, int index_1, int INPUT)
{
thrust::counting_iterator<int, thrust::device_space_tag> idxBegin(index_0);
thrust::counting_iterator<int, thrust::device_space_tag> idxEnd(index_1);
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

static void testHistogramParamThrust(int INPUT, int index_0, int index_1, bool print)
{
  test_sumfun2 mysumfun;
  thrust::equal_to<int> binary_pred;
  int nIndex = TESTMAXIDX;
  int N = index_1 - index_0;
  thrust::device_vector<int> keys_out(nIndex);
  thrust::device_vector<myTestType> vals_out(nIndex);
  thrust::device_vector<myTestType> h_vals_out(nIndex);
  thrust::device_vector<int> keys(N);
  thrust::device_vector<myTestType> values(N);
  initialize(keys, values, index_0, index_1, INPUT);
  thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
  thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), keys_out.begin(), vals_out.begin(), binary_pred, mysumfun);
  h_vals_out = vals_out;
  if (print)
  {
    printf("\nThrust results:\n");
    printf("vals = [ ");
    for (int i = 0; i < nIndex; i++)
    {
      myTestType tmp = h_vals_out[i];
        printf("(%d), ", tmp.real);
    }
    printf("]\n");
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
  printf("\t\t--stress\t Run the actual stress test - compares results to cpu results\n");
  printf("\t\t--print\t\t Print results of algorithm (check validity)\n");
  printf("\t\t--thrust\t Run on GPU but using thrust library\n");
}

int main (int argc, char** argv)
{
  int i;
  int index_0 = START_INDEX;
  int index_1 = index_0 + TEST_SIZE;
  int INPUT = 1;

  bool cpu = false;
  bool print = false;
  bool thrust = false;
  bool stress = false;

  printUsage();

  for (i = 0; i < argc; i++)
  {
    if (argv[i] && strcmp(argv[i], "--print") == 0)
      print = true;
    if (argv[i] && strcmp(argv[i], "--stress") == 0)
      stress = true;
    if (argv[i] && strcmp(argv[i], "--cpu") == 0)
      cpu = true;
    if (argv[i] && strcmp(argv[i], "--thrust") == 0)
      thrust = true;
  }
  for (INPUT = START_NBINS; INPUT < TESTMAXIDX; INPUT+=NBIN_INC)
  {
      for (i = 0; i < NRUNS; i++)
      {
        if (thrust)
        {
          #if ENABLE_THRUST
            testHistogramParamThrust(INPUT, index_0, index_1, print, stress);
          #else
            printf("\nTest was compiled without thrust support! Find 'ENABLE_THRUST' in source-code!\n\n Exiting...\n");
            break;
          #endif
        }
        else
        {
          bool success = testHistogramParam(INPUT, index_0, index_1, print, cpu, stress);
          if (!success) return 1;
        }
        print = false;
        // Run only once all stress-tests
        if (stress) break;
      }
      if (!stress) break;
  }
  return 0;
}
