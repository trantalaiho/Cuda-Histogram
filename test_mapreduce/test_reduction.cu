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
 *   Compile with:
 *
 *    nvcc -O4 -arch=<your_arch> -I../ test_reduction.cu -o test_reduction 
 *
 *   Optionally with -DTHRUST to include thrust support
 * 
 */

#define TEST_SIZE (32 * 1024 * 1024)   // 32 million inputs
#define NRUNS 1000          // Repeat 1000 times => 10 Gigainputs in total

#define MULTIDIM    3

#define NPERF_RUNS      20

#define TEST_MAX_VAL    100000

#ifdef THRUST
#define ENABLE_THRUST   1   // Enable thrust-based version also (xform-sort_by_key-reduce_by_key)
#else
#define ENABLE_THRUST   0   // Enable thrust-based version also (xform-sort_by_key-reduce_by_key)
#endif

#include "cuda_reduce.h"
#include "cuda_forall.h"
#include <cuda_runtime_api.h>

#if ENABLE_THRUST
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
/*#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>*/
#endif

#include <assert.h>
#include <stdio.h>



template <typename T>
struct test_xform2_multi
{
  __host__ __device__
  T operator() (T* input, int i, int multiIndex = 0) const {
    return input[i] * (multiIndex + 1);
  }
};

template <typename T>
struct test_xform2
{
  __host__ __device__
  T operator() (T* input, int i, int multiIndex = 0) const {
    return input[i];
  }
};




template <typename T>
struct pre_xformType
{
  __host__ __device__
  void operator() (T* input, int i, int multiIndex = 0) const {
    input[i] = input[i] * input[i];
  }
};



template <typename T>
struct test_xform3
{
  __host__ __device__
  T operator() (T x) const {
    return x;
  }
};

template <typename T>
struct test_sumfun2 {
  __device__ __host__
  T operator() (T res1, T res2) const{
    return res1 + res2;
  }
};


template <typename T>
struct test_minusfun2 {
  __device__ __host__
  T operator() (T res1, T res2) const{
    return res1 - res2;
  }
};




template <bool accurate, typename SCALART>
static SCALART getCPUres(SCALART* hostINPUT, int index_0, int index_1, bool xform, bool isFloatType, bool multi)
{
    SCALART cpures = 0;
    double tmpres = 0;
    volatile double comp = 0;
    test_sumfun2<SCALART> sumFun;
    test_xform2<SCALART> transformFun;
    test_xform2_multi<SCALART> multiXformFun;
    pre_xformType<SCALART> sqrFun;
    int nMulti = multi ? MULTIDIM : 1;
    if (isFloatType){
      int color;
      for (color = 0; color < nMulti; color++){
      for (int i = index_0; i < index_1; i++){ 
       if (!accurate){
         SCALART y;
         if (xform)
            sqrFun(hostINPUT, i);
         if (multi)
            y = multiXformFun(hostINPUT, i, color);
         else
            y = transformFun(hostINPUT, i, color);
         cpures = sumFun(cpures, y);
         tmpres = (double)cpures;
       }
       else {
          volatile double y, tmp;
          if (xform)
              sqrFun(hostINPUT, i);
          if (multi)
            y = (double)multiXformFun(hostINPUT, i, color) - comp;
          else
            y = (double)transformFun(hostINPUT, i, color) - comp;
          tmp = tmpres + y;
          comp = (tmp - tmpres) - y;
          tmpres = tmp;
          //printf("i = %d,  out_index = %d,  out_val = (%.3f, %.3f) \n",i, index, tmp.real, tmp.imag);
       }
      }
      }
      cpures = (SCALART)tmpres;
    }
    else
    {
      int color;
      for (color = 0; color < nMulti; color++){
        for (int i = index_0; i < index_1; i++){
          if (xform)
              sqrFun(hostINPUT, i);
          if (multi)
            cpures = sumFun(multiXformFun(hostINPUT, i, color), cpures);
          else
            cpures = sumFun(transformFun(hostINPUT, i, color), cpures);

        }
      }
    }
    return cpures;
}


static int ulpError(float val1, float val2)
{
  int mask1 = *((int*)(&val1));
  int mask2 = *((int*)(&val2));
  int result = mask1 - mask2;
  return result < 0 ? -result : result;
}


template <typename SCALART>
static double testReduce(SCALART* INPUT, SCALART* hostINPUT, int index_0, int index_1, bool print, bool cpurun, bool stress, bool xform, bool multi = false, bool kahan = false, double tolerance = 1e-12, bool isFloatType = false)
{
    double error = 0.0;
    SCALART initialval = 3;
    test_sumfun2<SCALART> sumFun;
    test_xform2<SCALART> transformFun;
    test_xform2_multi<SCALART> multiXformFun;
    pre_xformType<SCALART> sqrFun;
    test_minusfun2<SCALART> minusFun;
    int nMulti = multi ? MULTIDIM : 1;
    //test_indexfun2 indexFun;
    SCALART cpures = initialval, gpures = initialval;
    if (print)
        printf("\nTest reduce:\n\n");
    if (cpurun || stress){
        cpures += getCPUres<true>(hostINPUT, index_0, index_1, xform, isFloatType, multi);
        if (print){
            if (isFloatType)
                printf("CPU result: %f\n", (double)cpures);
            else
                printf("CPU result: %d\n", (int)cpures);
        }
    }

    if (1){
        if (cpurun){
          gpures += getCPUres<false>(hostINPUT, index_0, index_1, xform, isFloatType, multi);
        }
        else {
          if (xform)
              callTransformKernel(INPUT, sqrFun, index_0, index_1);
          if (kahan){
            SCALART zero = 0;
            //printf("\n\nKAHAN\n");
            if (multi)
              callKahanReduceKernel
                (INPUT, multiXformFun, sumFun, minusFun, index_0, index_1, &gpures, zero, nMulti);
            else
              callKahanReduceKernel
                (INPUT, transformFun, sumFun, minusFun, index_0, index_1, &gpures, zero, 1);
          }
          else
          {
            if (multi)
              callReduceKernel(INPUT, multiXformFun, sumFun, index_0, index_1, &gpures, nMulti);
            else
              callReduceKernel(INPUT, transformFun, sumFun, index_0, index_1, &gpures, 1);
          }
        }
        if (print){
            if (isFloatType)
                printf("GPU result: %f\n", (double)gpures);
            else
                printf("GPU result: %d\n", (int)gpures);
        }
        if (stress){
            bool pass = true;
            if (isFloatType)
            {
                double relTol = fabs(cpures) > fabs(gpures) ?   tolerance * fabs(cpures) :
                                                                tolerance * fabs(gpures);
                pass = fabs(cpures - gpures) <= relTol;
                error = fabs(cpures - gpures);
                if ((stress) && error != 0.0) { printf(" [err = %f,  ulp err = %d ] \n", error, ulpError(cpures, gpures)); fflush(stdout); }
            }
            else
            {
                pass = cpures == gpures;
            }
            if (!pass){
                if (isFloatType)
                    printf("GPU and CPU give different results! CPU: %f, GPU:%f, err = %g, with testsize: %d\n",
                            (double)cpures, (double)gpures, error, index_1 - index_0);
                else
                    printf("GPU and CPU give different results! CPU: %d, GPU:%d, with testsize: %d\n",
                            (int)cpures, (int)gpures, index_1 - index_0);
                //exit(1);
            }
        }
    }
    return error;
}


#define XSIZE   64
#define YSIZE   64
#define ZSIZE   32

//static cudaArray* input_cuArray;

texture<int, 2, cudaReadModeElementType> texIntRef;
texture<float, 2, cudaReadModeElementType> texFloatRef;
#define readTexI(x,y,z,t) tex2D(texIntRef, (float)(x + y * XSIZE), (float)(z+t*ZSIZE))
#define readTexF(x,y,z,t) tex2D(texFloatRef, (float)(x + y * XSIZE), (float)(z+t*ZSIZE))


static inline __host__ __device__
void indexTo4Size(int i, int size_out[4]){

  int tsize = i / (XSIZE * YSIZE * ZSIZE);
  if (tsize > 0){
    size_out[0] = XSIZE;
    size_out[1] = YSIZE;
    size_out[2] = ZSIZE;
    size_out[3] = tsize;
  }
  else
  {
    size_out[0] = 0;
    size_out[1] = 0;
    size_out[2] = 0;
    size_out[3] = 0;

  }
}

static inline __host__ __device__
int coord4ToIndex(int coords[4]){
  int i = coords[0] + coords[1] * XSIZE + coords[2]*YSIZE*XSIZE + coords[3]*YSIZE*XSIZE*ZSIZE;
  return i;
}


template <typename T>
struct test_xform4_multi_host
{
  T operator() (T* input, int* coords, int multiIndex = 0) const {
    return input[coord4ToIndex(coords)] * (multiIndex + 1);
  }
};


struct test_xform4_multi_int
{
  __device__
  int operator() (void* input, int* coords, int multiIndex = 0) const {
    return readTexI(coords[0], coords[1], coords[2], coords[3]) * (multiIndex + 1);
  }
};

struct test_xform4_multi_float
{
  __device__
  float operator() (void* input, int* coords, int multiIndex = 0) const {
    return readTexF(coords[0], coords[1], coords[2], coords[3]) * ((float)(multiIndex + 1));
  }
};


template <typename T>
struct test_xform4_host
{
  T operator() (T* input, int* coords, int multiIndex = 0) const {
    return input[coord4ToIndex(coords)];
    //return coords[0] + coords[1] + coords[2] - coords[3];
  }
};


struct test_xform4_int
{
  __device__
  int operator() (void* input, int* coords, int multiIndex = 0) const {
      //return coords[3];
      //return coords[0] + coords[1] + coords[2] - coords[3];
    return readTexI(coords[0], coords[1], coords[2], coords[3]);
  }
};

struct test_xform4_float
{
  __device__
  int operator() (void* input, int* coords, int multiIndex = 0) const {
    return readTexF(coords[0], coords[1], coords[2], coords[3]);
  }
};

template <typename T>
struct pre_xformType4
{
  __host__ __device__
  void operator() (T* input, int* coords, int multiIndex = 0) const {
    int i = coord4ToIndex(coords);
    input[i] = input[i] * input[i];
  }
};

/*
template <typename T>
struct test_xform4
{
  __host__ __device__
  T operator() (T* input, int* coords, int multiIndex = 0) const {
    //return input[coord4ToIndex(coords)];
      return input[coord4ToIndex(coords)];
  }
};


template <typename T>
struct pre_xformType4
{
  __host__ __device__
  void operator() (T* input, int* coords, int multiIndex = 0) const {
    int i = coord4ToIndex(coords);
    input[i] = input[i] * input[i];
  }
};
*/


template <typename SCALART>
static double testReduce4D(SCALART* INPUT, SCALART* hostINPUT, int index_0, int index_1, bool print, bool cpurun, bool stress, bool xform, bool multi = false, bool kahan = false, double tolerance = 1e-12, bool isFloatType = false)
{
    double error = 0.0;
    SCALART initialval = 3;
    test_sumfun2<SCALART> sumFun;
    test_xform4_host<SCALART> transformFunHost;
    test_xform4_int transformFunInt;
    test_xform4_float transformFunFloat;
    test_xform4_multi_host<SCALART> multiXformFunHost;
    test_xform4_multi_int multiXformFunInt;
    test_xform4_multi_float multiXformFunFloat;
    //test_xform4_multi<SCALART> multiXformFun;
    pre_xformType4<SCALART> sqrFun;
    test_minusfun2<SCALART> minusFun;
    int nMulti = multi ? MULTIDIM : 1;
    int x0s[4], x1s[4];
    indexTo4Size(index_0, x0s);
    indexTo4Size(index_1, x1s);

    int w = (x1s[0] - x0s[0]) * (x1s[1] - x0s[1]);
    int h = (x1s[2] - x0s[2]) * (x1s[3] - x0s[3]);
    // Bind textures:
    if (!cpurun){
        if (isFloatType)
        {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            // set texture parameters
            texFloatRef.normalized = false;        // access with normalized texture coordinates
            texFloatRef.filterMode = cudaFilterModePoint;
            texFloatRef.addressMode[0] = cudaAddressModeClamp;   // clamp texture coordinates
            texFloatRef.addressMode[1] = cudaAddressModeClamp;
            texFloatRef.addressMode[2] = cudaAddressModeClamp;
            //cudaBindTextureToArray(texFloatRef, input_cuArray, channelDesc);
            cudaBindTexture2D(NULL, texFloatRef, INPUT, channelDesc, w, h, w*4);
        }
        else
        {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int>();
            // set texture parameters
            texIntRef.normalized = false;        // access with normalized texture coordinates
            texIntRef.filterMode = cudaFilterModePoint;
            texIntRef.addressMode[0] = cudaAddressModeClamp;   // clamp texture coordinates
            texIntRef.addressMode[1] = cudaAddressModeClamp;
            texIntRef.addressMode[2] = cudaAddressModeClamp;
            //cudaBindTextureToArray(texFloatRef, input_cuArray, channelDesc);
            cudaBindTexture2D(NULL, texIntRef, INPUT, channelDesc, w, h, w*4);
        }
        {
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess){
               printf("Cudaerror = %s\n", cudaGetErrorString( error ));
               return -1.0;
            }
        }
    }
    //test_indexfun2 indexFun;
    SCALART cpures = initialval, gpures = initialval;
    if (print)
        printf("\nTest reduce:\n\n");
    if (cpurun || stress){
        int coords[4];
        double tmpres = 0;
        volatile double comp = 0;

        for (coords[3] = x0s[3]; coords[3] < x1s[3]; coords[3]++)
        for (coords[2] = x0s[2]; coords[2] < x1s[2]; coords[2]++)
        for (coords[1] = x0s[1]; coords[1] < x1s[1]; coords[1]++)
        for (coords[0] = x0s[0]; coords[0] < x1s[0]; coords[0]++)
        {
          int color;
          for (color = 0; color < nMulti; color++){
              if (xform)
                  sqrFun(hostINPUT, coords);
              if (isFloatType){
                volatile double y, tmp;
                if (multi)
                    y = (double)multiXformFunHost(hostINPUT, coords, color) - comp;
                else
                  y = (double)transformFunHost(hostINPUT, coords, color) - comp;
                tmp = tmpres + y;
                comp = (tmp - tmpres) - y;
                tmpres = tmp;
              }
              else
              {
                if (multi)
                  cpures = sumFun(multiXformFunHost(hostINPUT, coords, color), cpures);
                else
                  cpures = sumFun(transformFunHost(hostINPUT, coords, color), cpures);
              }
              //printf("i = %d,  out_index = %d,  out_val = (%.3f, %.3f) \n",i, index, tmp.real, tmp.imag);
          }
        }
        if (isFloatType)
          cpures = (SCALART)tmpres;
        if (print){
            if (isFloatType)
                printf("CPU result: %f\n", (double)cpures);
            else
                printf("CPU result: %d\n", (int)cpures);
        }
    }

    if (!cpurun){
        if (xform)
            callXformKernelNDim<4>(INPUT, sqrFun, x0s, x1s);
        if (kahan){
          SCALART zero = 0;
          //printf("\n\nKAHAN\n");
          if (multi){
            if (isFloatType)
                callKahanReduceKernelNDim<4>
                  (INPUT, multiXformFunFloat, sumFun, minusFun, x0s, x1s, &gpures, zero, nMulti);
            else
                callKahanReduceKernelNDim<4>
                  (INPUT, multiXformFunInt, sumFun, minusFun, x0s, x1s, &gpures, zero, nMulti);

          }
          else
          {
              if (isFloatType)
                  callKahanReduceKernelNDim<4>
                      (INPUT, transformFunFloat, sumFun, minusFun, x0s, x1s, &gpures, zero, 1);
              else
                  callKahanReduceKernelNDim<4>
                      (INPUT, transformFunInt, sumFun, minusFun, x0s, x1s, &gpures, zero, 1);

          }
        }
        else
        {
          if (multi){
              if (isFloatType)
                  callReduceKernelNDim<4>(INPUT, multiXformFunFloat, sumFun, x0s, x1s, &gpures, nMulti);
              else
                  callReduceKernelNDim<4>(INPUT, multiXformFunInt, sumFun, x0s, x1s, &gpures, nMulti);
          }
          else {
              if (isFloatType)
                  callReduceKernelNDim<4>(INPUT, transformFunFloat, sumFun, x0s, x1s, &gpures, 1);
              else
                  callReduceKernelNDim<4>(INPUT, transformFunInt, sumFun, x0s, x1s, &gpures, 1);
          }
        }
        if (print){
            if (isFloatType)
                printf("GPU result: %f\n", (double)gpures);
            else
                printf("GPU result: %d\n", (int)gpures);
        }
        if (stress){
            bool pass = true;
            if (isFloatType)
            {
                double relTol = fabs(cpures) > fabs(gpures) ?   tolerance * fabs(cpures) :
                                                                tolerance * fabs(gpures);
                pass = fabs(cpures - gpures) <= relTol;
                error = fabs(cpures - gpures);
                if ((stress) && error != 0.0) { printf(" [err = %f,  ulp err = %d ] \n", error, ulpError(cpures, gpures)); fflush(stdout); }
            }
            else
            {
                pass = cpures == gpures;
            }
            if (!pass){
                if (isFloatType)
                    printf("GPU and CPU give different results! CPU: %f, GPU:%f, err = %g, with testsize: %d\n",
                            (double)cpures, (double)gpures, error, index_1 - index_0);
                else
                    printf("GPU and CPU give different results! CPU: %d, GPU:%d, with testsize: %d\n",
                            (int)cpures, (int)gpures, index_1 - index_0);
                exit(1);
            }
        }
    }
    if (isFloatType)
        cudaUnbindTexture(texFloatRef);
    else
        cudaUnbindTexture(texIntRef);
    return error;
}




#if ENABLE_THRUST

// NOTE: Take advantage here of the fact that this is the classical histogram with all values = 1
// And also that we know before hand the number of indices coming out
template <typename SCALART>
static double testReduceThrust(SCALART* INPUT, SCALART* hostINPUT, int index_0, int index_1, bool print, bool useXFormReduce, bool stress , double tolerance = 1e-12, bool isFloatType = false)
{
    double error = 0.0;
    test_sumfun2<SCALART> mysumfun;
    test_xform3<SCALART> xformFun;
    int N = index_1 - index_0;
    SCALART init = 3;
    SCALART cpures = init;
    SCALART t_res;
    thrust::device_ptr<SCALART> vals(INPUT);
    // Reduce
    if (useXFormReduce)
        t_res = thrust::transform_reduce(vals, vals + N, xformFun, init, mysumfun);
    else
        t_res = thrust::reduce(vals, vals + N, init, mysumfun);

    if (stress){
        cpures += getCPUres<true>(hostINPUT, index_0, index_1, useXFormReduce, isFloatType, false);
        bool pass = true;
        if (isFloatType)
        {
            double relTol = fabs(cpures) > fabs(t_res) ?   tolerance * fabs(cpures) :
                                                            tolerance * fabs(t_res);
            pass = fabs(cpures - t_res) <= relTol;
            error = fabs(cpures - t_res);
            if ((stress) && error != 0.0) { printf(" [err = %f,  ulp err = %d ] \n", error, ulpError(cpures, t_res)); fflush(stdout); }
        }
        else
        {
            pass = cpures == t_res;
        }
        if (!pass){
            if (isFloatType)
                printf("Thrust and CPU give different results! CPU: %f, GPU:%f, err = %g, with testsize: %d\n",
                        (double)cpures, (double)t_res, error, index_1 - index_0);
            else
                printf("Thrust and CPU give different results! CPU: %d, GPU:%d, with testsize: %d\n",
                        (int)cpures, (int)t_res, index_1 - index_0);
            exit(1);
        }
    }
    if (print){
        if (isFloatType)
            printf("Thrust result: %f\n", (double)t_res);
        else
            printf("Thrust result: %d\n", (int)t_res);
    }
    return error;
}
#endif

void printUsage(void)
{
  printf("\n");
  printf("Test reduction algorithm\n\n");
  printf("\tOptions:\n\n");
  printf("\t\t--cpu\t\t Run on CPU serially instead of GPU\n");
  printf("\t\t--print\t\t Print result of algorithm (check validity)\n");
  printf("\t\t--kahan\t\t Run on GPU using (mostly) Kahan summation\n");
  printf("\t\t--thrust\t Run on GPU but using thrust library\n");
  printf("\t\t--thrust_xform\t Use thrusts xform-reduce instead of reduce\n");


  printf("\t\t--load\t\t Use 32-bit texture data\n");
  printf("\t\t--rnd\t\t Take uniform random vals\n");
  printf("\t\t--stress\t Run some stress-tests\n");
  printf("\t\t--float\t\t Use floating point numbers in reduction (order dependence)\n");
  printf("\t\t--multi\t\t Test multi-reduce \n");
  printf("\t\t--4dim\t\t Test 4-D reduce \n");
  printf("\t\t--megastress\t Run all stress-tests (all sizes from 0 to TEST_SIZE)\n");
  printf("\t\t--perf\t\t Do perf-runs\n");
  printf("\t\t--xform\t\t Do a transform using cuda_forall.h before reduction.\n");
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




static inline int getInput(int i, unsigned int* texData, int dataSize, bool rnd, bool isFloat = false)
{
  if (texData)
  {
    static int index = i % dataSize;
    static int round = 0;
    unsigned int val = texData[index];
    int result;

    if ((i & 0x1) == 0)
    {
        result = ((0xffff0000 & val) >> 16) + round;
    }
    else
    {
        result = (0x0000ffff & val) + round;
        index++;
        if (index >= dataSize)
        {
            index = 0;
            round += 7;
        }
    }
    result = (int)(result % (TEST_MAX_VAL));
    if (isFloat){
        float tmp = (float)result / (float)(TEST_MAX_VAL);
        int* tmpptr = (int*)(&tmp);
        result = *tmpptr;
    }
    return result;
  }
  else
  {
    static unsigned int current = 0xf1232345;
    const unsigned int mult = 1664525;
    const unsigned int add = 1013904223ul;

    current = current * mult + add;
    if (!rnd)
        current = i / 100;
    i = (int)(current % (TEST_MAX_VAL * 1024));
    if (isFloat){
        float tmp = (float)(i*10) / (float)(TEST_MAX_VAL * 1024);
        tmp -= 4.0;
        int* tmpptr = (int*)(&tmp);
        i = *tmpptr;
    }
    else{
      i /= 1024;
    }
    return i;
  }
}

static void fillInput(int* input, bool load, bool rnd, bool isFloat = false)
{
  int i;
  unsigned int* texData = NULL;
  int dataSize = 0;
  if (load && !rnd)
  {
      texData = MyTexture_load("texture.raw", &dataSize);
  }
  for (i = 0; i < TEST_SIZE;)
  {
    *input++ = getInput(i++, texData, dataSize, rnd, isFloat);
    *input++ = getInput(i++, texData, dataSize, rnd, isFloat);
    *input++ = getInput(i++, texData, dataSize, rnd, isFloat);
    *input++ = getInput(i++, texData, dataSize, rnd, isFloat);
  }
  if (texData) free(texData);
}


int main (int argc, char** argv)
{
  int i, perfrun;
  int index_0 = 0;
  int index_1 = index_0 + TEST_SIZE;

  bool cpu = false;
  bool print = false;
  bool thrust = false;
  bool xform = false;
  bool stress = false;
  bool mega = false;
  bool perf = false;
  bool prexform = false;
  bool dofloat = false;

  bool rnd = false;
  bool load = false;
  bool kahan = false;
  bool multi = false;
  bool fourD = false;

  double maxerr = 0.0;
  int errindex = 0;

  printUsage();

  for (i = 0; i < argc; i++)
  {
    if (argv[i] && strcmp(argv[i], "--cpu") == 0)
      cpu = true;
    if (argv[i] && strcmp(argv[i], "--print") == 0)
      print = true;
    if (argv[i] && strcmp(argv[i], "--thrust") == 0)
      thrust = true;
    if (argv[i] && strcmp(argv[i], "--thrust_xform") == 0)
      xform = true;
    if (argv[i] && strcmp(argv[i], "--stress") == 0)
      stress = true;
    if (argv[i] && strcmp(argv[i], "--megastress") == 0)
      mega = true;
    if (argv[i] && strcmp(argv[i], "--float") == 0)
        dofloat = true;
    if (argv[i] && strcmp(argv[i], "--load") == 0)
      load = true;
    if (argv[i] && strcmp(argv[i], "--rnd") == 0)
        rnd = true;
    if (argv[i] && strcmp(argv[i], "--perf") == 0)
        perf = true;
    if (argv[i] && strcmp(argv[i], "--xform") == 0)
        prexform = true;
    if (argv[i] && strcmp(argv[i], "--kahan") == 0)
        kahan = true;
    if (argv[i] && strcmp(argv[i], "--multi") == 0)
      multi = true;
    if (argv[i] && strcmp(argv[i], "--4dim") == 0)
      fourD = true;
  }
  if (xform) thrust = true;
  {
    // Allocate keys:
    int* INPUT = NULL;
    int* hostINPUT = (int*)malloc(sizeof(int) * (TEST_SIZE + 3));
    assert(hostINPUT);
    fillInput(hostINPUT, load, rnd, dofloat);
    if (!cpu)
    {
      cudaMalloc(&INPUT, sizeof(int) * TEST_SIZE);
      assert(INPUT);
      cudaMemcpy(INPUT, hostINPUT, sizeof(int) * TEST_SIZE, cudaMemcpyHostToDevice);
    }
    // Create events for timing:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (stress) printf("Running stress-tests...\n");

    // Now start timer - we run on stream 0 (default stream):
    for (perfrun = 0; perfrun < (perf ? NPERF_RUNS : 1); perfrun++)
    {
        int l_limit = index_1;
        int nruns = mega ? TEST_SIZE : NRUNS;
        cudaEventRecord(start, 0);
        if (perf) l_limit = (perfrun+1) * TEST_SIZE / NPERF_RUNS;

        for (i = 0; i < nruns; i++)
        {

          if (mega){
            l_limit = i;
          }
          else if (stress) {
              if (i < NRUNS / 5)
                  l_limit = index_0 + i - 1;
              else if (i > NRUNS - NRUNS / 5)
                  l_limit = index_0 + TEST_SIZE - (NRUNS - 1- i);
              else l_limit = (int)((long long)index_0 + (long long)i * ((long long)TEST_SIZE / (long long)NRUNS));
              if (l_limit >= TEST_SIZE)
                l_limit = TEST_SIZE -1;
          }
          if (thrust)
          {
            if (multi) printf("\nSorry - thrust version doesn't support yet multi!...\n");
            #if ENABLE_THRUST
            if (dofloat)
            {
                double error =
                    (double)testReduceThrust(
                        (float*)INPUT, (float*)hostINPUT, index_0, l_limit,
                        print, xform, stress || mega, 1e-6, true);
                if (error > maxerr){ maxerr = error; errindex = l_limit - index_0; }
            }
            else
            {
              testReduceThrust(INPUT, hostINPUT, index_0, l_limit, print, xform, stress || mega);
            }
              //testReduceThrust(INPUT, hostINPUT, index_0, l_limit, print, xform, stress || mega);
            #else
              printf("\nTest was compiled without thrust support! Find 'ENABLE_THRUST' in source-code!\n\n Exiting...\n");
              break;
            #endif
          }
          else
          {
              if (dofloat)
              {
                  double error =
                      (double)testReduce(
                          (float*)INPUT, (float*)hostINPUT, index_0, l_limit,
                          print, cpu, stress || mega, prexform, multi, kahan, 1e-6, true);
                  if (error > maxerr){ maxerr = error; errindex = l_limit - index_0; }
              }
              else
              {
                if (fourD)
                  testReduce4D(INPUT, hostINPUT, index_0, l_limit, print, cpu, stress || mega, prexform, multi, kahan);
                else
                  testReduce(INPUT, hostINPUT, index_0, l_limit, print, cpu, stress || mega, prexform, multi, kahan);
              }
          }
          if (stress || (mega && i % 100 == 0)) { printf("%d \n", l_limit - index_0); fflush(stdout); }

          if (!stress) print = false;
        }

        {
            float t_ms;
            cudaEventRecord(stop, 0);
            cudaThreadSynchronize();
            cudaEventElapsedTime(&t_ms, start, stop);
            double t = t_ms * 0.001f;
            double GKps = (((double)(l_limit - index_0) * (double)NRUNS)) / (t*1.e9);
            if (multi) GKps *= (double)MULTIDIM;
            if (perf){
                static bool first = true;
                if (first) {
                    printf("Reduction input size, Runtime in loops(s), Thoughput (Gkeys/s):\n");
                    first = false;
                }
                printf("%d\t\t%fs\t%3f GK/s \n", l_limit - index_0, t, GKps);
            }
            else if (!stress)
                printf("Runtime in loops: %fs, Thoughput (Gkeys/s): %3f GK/s \n", t, GKps);
            if (dofloat)
                printf("max error(%d) = %g\n", errindex, maxerr);
        }
    }
    if (INPUT) cudaFree(INPUT);
    if (hostINPUT) free(hostINPUT);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  return 0;
}

