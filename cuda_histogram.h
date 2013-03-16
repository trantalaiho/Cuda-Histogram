/*
 * cuda_histogram.h
 *
 *  Created on: 3.5.2011
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
 *
 */

#ifndef CUDA_HISTOGRAM_H_
#define CUDA_HISTOGRAM_H_

#include <cuda_runtime_api.h>


// Public API:

/*------------------------------------------------------------------------*//*!
 * \brief   Enumeration to define the type of histogram we are interested in
 *//**************************************************************************/

enum histogram_type {
  histogram_generic,        /*!< \brief Generic histogram, for any types */
  histogram_atomic_inc,     /*!< \brief Each output-value is constant 1 */
  histogram_atomic_add,     /*!< \brief Output-type is such that atomicAdd()
                                        function can be used */
};


/*------------------------------------------------------------------------*//*!
 * \brief   Check the size of the temporary buffer needed for histogram
 *
 *  This function can be used to check the size of the temporary buffer7
 *  needed in \ref callHistogramKernel() so that a buffer of correct
 *  size can be passed in to the histogram call -- this way one can
 *  avoid the latency involved in GPU-memory allocations when running
 *  multiple histograms
 *
 * \tparam  histotype   Type of histogram: generic, atomic-inc or atomic-add
 * \tparam  OUTPUTTYPE  The type for the outputs of the histogram
 *                      (type of each bin)
 * \param   zero        Invariant element of the sum-function
 *                      (TODO: remove this?)
 * \param   nOut        Number of bins in the histogram
 *//**************************************************************************/

template <histogram_type histotype, typename OUTPUTTYPE>
static
int
getHistogramBufSize(OUTPUTTYPE zero, int nOut);



/*------------------------------------------------------------------------*//*!
 * \brief   Implements generic histogram on the device
 *
 * \tparam  histotype   Type of histogram: generic, atomic-inc or atomic-add
 * \tparam  nMultires   Number of results each input produces. Typically 1
 * \tparam  INDEXT      Integer type to be used for indexing (int, size_t, ...)
 * \tparam  INPUTTYPE   Arbitrary type passed in as input
 * \tparam  TRANSFORMFUNTYPE    Function object that takes in the input, index
 *                              and produces 'nMultires' results and indices.
 *                              The signature therefore is as follows:
 *          struct xform
 *          {
 *              __device__
 *              void operator() (
 *                  INPUTTYPE* input, int index, int* result_index,
 *                  OUTPUTTYPE* results, int nresults) const
 *              {
 *                  *result_index[0] = <Some bin-number - from 0 to nOut - 1>;
 *                  *results = input[i];
 *              }
 *          };
 * \note    Passing out indices out of range (of [0, nOut-1]) will result
 *          in unspecified behavior
 * \tparam  SUMFUNTYPE  Function object type that implements the binary
 *                      reduction of two input variables
 * \tparam  OUTPUTTYPE  The type for the outputs of the histogram
 *
 * \param   input       Arbitrary input that will be passed as context
 *                      for the transform-operator
 * \param   xformObj    The transform-function object for this operation
 * \param   sumfunObj   The binary-operator that combines OUTPUTTYPE-values
 *                      inside each bin.
 * \param   start       Starting index for the operation - first index
 *                      to be passed into the transform-functor
 * \param   end         Ending index for the operation - first index
 *                      _NOT_ to be passed into the transform-functor
 * \param   zero        Invariant object of the sumfunObj-functor (0 + x = x)
 * \param   out         Output-bins - this operations adds on top of the values
 *                      already contained on bins
 * \param   nOut        Number of arrays in output and therefore number of bins
 * \param   outInDev    The output-bins \ref out reside in device (ie GPU) memory
 *                      Default value: false
 * \param   stream      Cudastream to be used for this operation - all GPU
 *                      operations will be run on this stream. \note in case the
 *                      output-bins reside in HOST memory the last memory copy
 *                      from device to host will be synchronized, so that when
 *                      this API-call returns, the results will be available on
 *                      host memory (ie. no cudaStreamSynchronize() is necessary)
 *                      Default value: 0 - the default stream
 * \param   tmpBuffer   A temporary buffer of device (GPU) memory to be used.
 *                      Has to be large enough - check size with
 *                      \ref getHistogramBufSize().
 *//**************************************************************************/

template <histogram_type histotype, int nMultires,
            typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
            typename OUTPUTTYPE, typename INDEXT>
static
cudaError_t
callHistogramKernel(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero, OUTPUTTYPE* out, int nOut,
    bool outInDev = false,
    cudaStream_t stream = 0, void* tmpBuffer = NULL,
    bool allowMultiPass = true);



/*------------------------------------------------------------------------*//*!
 * \brief   Histogram implementation with multidimensional input indices
 *
 *  Exactly as \ref callHistogramKernel(), except that input-indices
 *  are replaced by a multidimensional array of indices
 *
 * \tparam  nDim        Number of dimesions in multidimensional indices
 * \param   starts      Array of length nDim, giving starting index for
 *                      each dimension of input indices (exclusive (*))
 * \param   ends        Array of length nDim, giving ending index for
 *                      each dimension of input indices (inclusive)
 * \note    (*) Indices are the [start, start + size - 1], size =  end - start
 *//**************************************************************************/

template <histogram_type histotype, int nMultires, int nDim,
            typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
            typename OUTPUTTYPE, typename INDEXT>
static
cudaError_t
callHistogramKernelNDim(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT* starts, INDEXT* ends,
    OUTPUTTYPE zero, OUTPUTTYPE* out, int nOut,
    bool outInDev = false,
    cudaStream_t stream = 0, void* tmpBuffer = NULL,
    bool allowMultiPass = true);


/*------------------------------------------------------------------------*//*!
 * \brief   Histogram implementation with two-dimensional input indices
 *
 *  Exactly as \ref callHistogramKernel(), except that input-indices
 *  are replaced by a two indices
 *
 * \param   x0          Start x-coordinate
 * \param   x1          End x-coordinate (width = x1 - x0)
 * \param   y0          Start y-coordinate
 * \param   y1          End y-coordinate (height = y1 - y0)
 *//**************************************************************************/

template <histogram_type histotype, int nMultires,
            typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
            typename OUTPUTTYPE, typename INDEXT>
cudaError_t
callHistogramKernel2Dim(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT x0, INDEXT x1,
    INDEXT y0, INDEXT y1,
    OUTPUTTYPE zero, OUTPUTTYPE* out, int nOut,
    bool outInDev,
    cudaStream_t stream, void* tmpBuffer,
    bool allowMultiPass = true);




// Start implementation:


// Define this to be one to enable cuda Error-checks (runs a tad slower)
#define H_ERROR_CHECKS      0

#if H_ERROR_CHECKS
    #include <assert.h>
    #include <stdio.h>
#endif

#define HBLOCK_SIZE_LOG2    7
#define HBLOCK_SIZE         (1 << HBLOCK_SIZE_LOG2) // = 32

#define HMBLOCK_SIZE_LOG2   8
#define HMBLOCK_SIZE         (1 << HMBLOCK_SIZE_LOG2) // = 32


#define LBLOCK_SIZE_LOG2    5
#define LBLOCK_SIZE         (1 << LBLOCK_SIZE_LOG2) // = 256
#define LBLOCK_WARPS        (LBLOCK_SIZE >> 5)

#define USE_MEDIUM_PATH         1

#if USE_MEDIUM_PATH
// For now only MEDIUM_BLOCK_SIZE_LOG2 == LBLOCK_SIZE_LOG2 works
#   define MEDIUM_BLOCK_SIZE_LOG2   8
#   define MEDIUM_BLOCK_SIZE        (1 << MEDIUM_BLOCK_SIZE_LOG2) // 128
#   define MBLOCK_WARPS             (MEDIUM_BLOCK_SIZE >> 5)
#define MED_THREAD_DEGEN    16
#endif

#define RBLOCK_SIZE         64
#define RMAXSTEPS           80
#define NHSTEPSPERKEY       32
#define MAX_NHSTEPS         1024
#define MAX_MULTISTEPS      1024
#define MAX_NLHSTEPS        2048

#define GATHER_BLOCK_SIZE_LOG2  6
#define GATHER_BLOCK_SIZE       (1 << GATHER_BLOCK_SIZE_LOG2)

#define STRATEGY_CHECK_INTERVAL_LOG2    7
#define STRATEGY_CHECK_INTERVAL         (1 << STRATEGY_CHECK_INTERVAL_LOG2)

#define HISTOGRAM_DEGEN_LIMIT   20


#define HASH_COLLISION_STEPS    2

const int numActiveUpperLimit = 24;

#define USE_JENKINS_HASH 0

#define LARGE_NBIN_CHECK_INTERVAL_LOG2  5
#define LARGE_NBIN_CHECK_INTERVAL       (1 << LARGE_NBIN_CHECK_INTERVAL_LOG2)

#define SMALL_BLOCK_SIZE_LOG2   6
#define SMALL_BLOCK_SIZE        (1 << SMALL_BLOCK_SIZE_LOG2)
#define MAX_SMALL_STEPS         2040


#if __CUDA_ARCH__ >= 120
#define USE_ATOMICS_HASH    0
#else
#define USE_ATOMICS_HASH    0
#endif




#if (__CUDA_ARCH__ >= 200)
#   define USE_BALLOT_HISTOGRAM    1
#else
#   define USE_BALLOT_HISTOGRAM    0
#endif

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __shared__
#define __shared__
#endif



//#include <stdio.h>

template <typename OUTPUTTYPE, typename SUMFUNTYPE>
__global__
void multireduceKernel(OUTPUTTYPE* input, int n, int nOut, int nsteps, SUMFUNTYPE sumFun, OUTPUTTYPE zero, int stride, OUTPUTTYPE* initialValues)
{
  int tid = threadIdx.x;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;
  OUTPUTTYPE myout = zero;
  int i;
  for (i = 0; i < nsteps; i++)
  {
    int subIndex = bidx * RBLOCK_SIZE + tid;
    int cidx = subIndex + i * RBLOCK_SIZE * gridDim.x;
    if (cidx < n)
    {
//       printf("t(%2d)b(%3d,%2d) r(%d)\n", tid, bidx, bidy, cidx + bidy * stride);
       myout = sumFun(myout, input[cidx + bidy * stride]);
    }
  }
  __shared__ OUTPUTTYPE tmp[RBLOCK_SIZE / 2];
  for (int curLimit = RBLOCK_SIZE / 2; curLimit > 0; curLimit >>= 1)
  {
    // First write out the current result for threads above the limit
    if (tid >= curLimit && tid < (curLimit << 1))
      tmp[tid - curLimit] = myout;
    // Otherwise wait for the write the complete and add that value to our result
    __syncthreads();
    if (tid < curLimit)
      myout = sumFun(myout, tmp[tid]);
    // IMPORTANT: Wait before new loop for the read to complete
    __syncthreads();
  }
  // Done! myout contains the result for our block for thread 0!!
  if (tid == 0)
  {
    // NOTE: If gridDim == 1 then we have finally reached the last iteration and
    // can write the result into the final result-value array
    // (ie. The same as initialvalue-array)
    if (gridDim.x == 1)
    {
      OUTPUTTYPE initVal = initialValues[bidy];
      initialValues[bidy] = sumFun(initVal, myout);
      // And we are DONE!
    }
    else
    {
 //     printf("t(%2d)b(%3d,%2d) w(%d)\n", tid, bidx, bidy, bidx + bidy * stride);
      initialValues[bidx + bidy * stride] = myout;
    }
  }
}

/*------------------------------------------------------------------------*//*!
 * \brief   Implements multidimensional reduction on the device
 *
 * \tparam  OUTPUTTYPE  The type for the inputs and outputs of the reduction
 * \tparam  SUMFUNTYPE  Function object type that implements the binary
 *                      reduction of two input variables
 *
 * \param   arrLen      The length of each array in inputs
 * \param   nOut        Number of arrays in input
 * \param   input       The input data laid out as a two-dimensional array,
 *                      where the 'n:th' element of 'i:th' array
 *                      is input[n + i * arrLen].
 *                      MUST RESIDE IN DEVICE MEMORY!
 * \param   h_results   Pointer to an array. where the results
 *                      are to be stored. 'n:th' result will be located at
 *                      h_results[n]. MUST RESIDE IN HOST-MEMORY!
 *
 * \note    \ref input has to reside in device memory
 * \note    \ref h_results has to reside in host memory
 * \note    \ref SUMFUNTYPE has to implement operator:
 *          '__device__ OUTPUTTYPE operator()(OUTPUTTYPE in1, OUTPUTTYPE in2)'
 *
 * \note    Uses \ref RBLOCK_SIZE and \ref RMAX_STEPS as parameters for
 *          kernel-sizes. Varying these parameters can affect performance.
 *//**************************************************************************/

template <typename OUTPUTTYPE, typename SUMFUNTYPE>
static
void callMultiReduce(
    int arrLen, int nOut, OUTPUTTYPE* h_results, OUTPUTTYPE* input,
    SUMFUNTYPE sumFunObj, OUTPUTTYPE zero,
    cudaStream_t stream, void* tmpbuf, bool outInDev)
{
  int n = arrLen;
  // Set-up yet another temp buffer: (TODO: Pool alloc somehow?)
  OUTPUTTYPE* resultTemp = NULL;
  // TODO: Why do we need such a large temporary array?
  // Shouldn't sizeof(OUTPUTTYPE) * nOut * xblocks be enough??
  if (tmpbuf)
  {
    resultTemp = (OUTPUTTYPE*)tmpbuf;
  }
  else
  {
    cudaMalloc((void**)&resultTemp, sizeof(OUTPUTTYPE) * nOut * arrLen);
#if H_ERROR_CHECKS
    //printf("resultTemp = %p\n", resultTemp);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("Cudaerror0 = %s\n", cudaGetErrorString( error ));
#endif
  }
  OUTPUTTYPE* output = resultTemp;
  enum cudaMemcpyKind fromOut = outInDev ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
  enum cudaMemcpyKind toOut = outInDev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

  // Copy initial values:
  do
  {
    int steps = (n + (RBLOCK_SIZE - 1)) / RBLOCK_SIZE;
    if (steps > RMAXSTEPS)
      steps = RMAXSTEPS;
    int yblocks = nOut;
    int xblocks = (n + (steps * RBLOCK_SIZE - 1)) / (steps * RBLOCK_SIZE);

    const dim3 block = RBLOCK_SIZE;
    const dim3 grid(xblocks, yblocks, 1);


    if (xblocks == 1) // LAST ONE to start
    {
      //printf("cudaMemcpy(%p, %p, %d, %d);\n", output, h_results, sizeof(OUTPUTTYPE) * nOut, fromOut);
      if (stream != 0)
        cudaMemcpyAsync(output, h_results, sizeof(OUTPUTTYPE) * nOut, fromOut, stream);
      else
        cudaMemcpy(output, h_results, sizeof(OUTPUTTYPE) * nOut, fromOut);
    }
#if H_ERROR_CHECKS
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("Cudaerror1 = %s\n", cudaGetErrorString( error ));
#endif
    // Then the actual kernel call
    multireduceKernel<<<grid, block, 0, stream>>>(input, n, nOut, steps, sumFunObj, zero, arrLen, output);
#if H_ERROR_CHECKS
    error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("Cudaerror2 = %s\n", cudaGetErrorString( error ));
#endif
    if (xblocks > 1)
    {
        // Swap pointers:
        OUTPUTTYPE* tmpptr = output;
        output = input;
        input = tmpptr;
    }

    n = xblocks;
  } while(n > 1);
  // Then copy back the results:
  //cudaMemcpyAsync(h_results, resultTemp, sizeof(OUTPUTTYPE) * nOut, cudaMemcpyDeviceToHost, CURRENT_STREAM());
  // TODO: Support async copy here??
  if (outInDev && stream != 0)
    cudaMemcpyAsync(h_results, output, sizeof(OUTPUTTYPE) * nOut, toOut, stream);
  else
    cudaMemcpy(h_results, output, sizeof(OUTPUTTYPE) * nOut, toOut);
#if H_ERROR_CHECKS
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("Cudaerror3 = %s\n", cudaGetErrorString( error ));
#endif
  if (!tmpbuf)
  {
    cudaFree(resultTemp);
  }
#if H_ERROR_CHECKS
    error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("Cudaerror4 = %s\n", cudaGetErrorString( error ));
#endif
}


template <typename SUMFUNTYPE, typename OUTPUTTYPE>
__global__
void gatherKernel(SUMFUNTYPE sumfunObj, OUTPUTTYPE* blockOut, int nOut, int nEntries, OUTPUTTYPE zero)
{
    //int resIdx = threadIdx.x + blockDim.x * blockIdx.x;
    int resIdx = blockIdx.x;
    if (resIdx < nOut)
    {
        // Let's divide the nEntries first evenly on all threads and read 4 entries in a row
        int locEntries = (nEntries) >> (GATHER_BLOCK_SIZE_LOG2);

        // Note: Original array entry is stored in resIdx + nOut * nEntries!
        OUTPUTTYPE res = zero;
        if (threadIdx.x == 0)
            res = blockOut[resIdx + nOut * nEntries];

        // Shift starting ptr:
        blockOut = &blockOut[resIdx];
        int locIdx = threadIdx.x * locEntries;
        for (int i=0; i < locEntries/4; i++)
        {

          OUTPUTTYPE x1 = blockOut[nOut * (locIdx + (i << 2))];
          OUTPUTTYPE x2 = blockOut[nOut * (locIdx + (i << 2) + 1)];
          OUTPUTTYPE x3 = blockOut[nOut * (locIdx + (i << 2) + 2)];
          OUTPUTTYPE x4 = blockOut[nOut * (locIdx + (i << 2) + 3)];
          res = sumfunObj(res, x1);
          res = sumfunObj(res, x2);
          res = sumfunObj(res, x3);
          res = sumfunObj(res, x4);
        }
        // Then do the rest
        for (int j = (locEntries/4)*4; j < locEntries; j++)
        {
          OUTPUTTYPE x1 = blockOut[nOut * (locIdx + j)];
          res = sumfunObj(res, x1);
        }
        // Still handle rest starting from index "locEntries * BLOCK_SIZE":
        locIdx = threadIdx.x + (locEntries << GATHER_BLOCK_SIZE_LOG2);
        if (locIdx < nEntries)
          res = sumfunObj(res, blockOut[nOut * locIdx]);
        // Ok - all that is left is to do the final parallel reduction between threads:
        {
            __shared__  OUTPUTTYPE data[GATHER_BLOCK_SIZE];
            //volatile OUTPUTTYPE* data = (volatile OUTPUTTYPE*)&dataTmp[0];
            // TODO Compiler complains with volatile from this - why?
            //error: no operator "=" matches these operands
              //         operand types are: volatile myTestType_s = myTestType
            // Silly - does not happen with built-in types (nice...)
            data[threadIdx.x] = res;
        #if GATHER_BLOCK_SIZE == 512
            __syncthreads();
            if (threadIdx.x < 256)
              data[threadIdx.x] = sumfunObj(data[threadIdx.x], data[threadIdx.x + 256]);
        #endif
#if GATHER_BLOCK_SIZE >= 256
          __syncthreads();
          if (threadIdx.x < 128)
            data[threadIdx.x] = sumfunObj(data[threadIdx.x], data[threadIdx.x + 128]);
#endif
#if GATHER_BLOCK_SIZE >= 128
          __syncthreads();
          if (threadIdx.x < 64)
            data[threadIdx.x] = sumfunObj(data[threadIdx.x], data[threadIdx.x + 64]);
          __syncthreads();
#endif
#if GATHER_BLOCK_SIZE >= 64
          __syncthreads();
          if (threadIdx.x < 32)
            data[threadIdx.x] = sumfunObj(data[threadIdx.x], data[threadIdx.x + 32]);
#endif
          __syncthreads();
          if (threadIdx.x < 16) data[threadIdx.x] = sumfunObj(data[threadIdx.x], data[threadIdx.x + 16]);
          __syncthreads();
          if (threadIdx.x < 8) data[threadIdx.x] = sumfunObj(data[threadIdx.x], data[threadIdx.x + 8]);
          __syncthreads();
          if (threadIdx.x < 4) data[threadIdx.x] = sumfunObj(data[threadIdx.x], data[threadIdx.x + 4]);
          __syncthreads();
          if (threadIdx.x < 2) data[threadIdx.x] = sumfunObj(data[threadIdx.x], data[threadIdx.x + 2]);
          __syncthreads();
          if (threadIdx.x < 1) *blockOut = sumfunObj(data[threadIdx.x], data[threadIdx.x + 1]);
        }
    }
}

#define FREE_MUTEX_ID   0xffeecafe
#define TAKE_WARP_MUTEX(ID) do { \
                                int warpIdWAM = threadIdx.x >> 5; \
                                __shared__ volatile int lockVarWarpAtomicMutex;\
                                bool doneWAM = false;\
                                bool allDone = false; \
                                    while(!allDone){ \
                                        __syncthreads(); \
                                        if (!doneWAM) lockVarWarpAtomicMutex = warpIdWAM; \
                                        __syncthreads(); \
                                        if (lockVarWarpAtomicMutex == FREE_MUTEX_ID) allDone = true; \
                                        __syncthreads(); \
                                        if (lockVarWarpAtomicMutex == warpIdWAM){ /* We Won */

                                            // User code comes here

#define GIVE_WARP_MUTEX(ID)                 doneWAM = true; \
                                            lockVarWarpAtomicMutex = FREE_MUTEX_ID; \
                                        } \
                                    } \
                                    __syncthreads(); \
                            } while(0)

// NOTE: Init must be called from divergent-free code (or with exited warps)
#define INIT_WARP_MUTEX2(MUTEX) do { MUTEX = FREE_MUTEX_ID; __syncthreads(); } while(0)
#if 0 && __CUDA_ARCH__ >= 120 // TODO: NOT WORKING THIS CODEPATH - find out why
#define TAKE_WARP_MUTEX2(MUTEX) do { \
                                int warpIdWAM = 1000000 + threadIdx.x / 32; \
                                bool doneWAM = false;\
                                    while(!doneWAM){ \
                                        int old = -2; \
                                        if (threadIdx.x % 32 == 0) \
                                            old = atomicCAS(&MUTEX, FREE_MUTEX_ID, warpIdWAM); \
                                        if (__any(old == FREE_MUTEX_ID)){ /* We Won */

                                            // User code comes here

#define GIVE_WARP_MUTEX2(MUTEX)             doneWAM = true; \
                                            atomicExch(&MUTEX, FREE_MUTEX_ID); \
                                        } \
                                    } \
                            } while(0)
#else
#define TAKE_WARP_MUTEX2(MUTEX) do { \
                                int warpIdWAM = 1000000 + threadIdx.x / 32; \
                                bool doneWAM = false;\
                                bool allDone = false; \
                                    while(!allDone){ \
                                        __syncthreads(); \
                                        if (!doneWAM) MUTEX = warpIdWAM; \
                                        __syncthreads(); \
                                        if (MUTEX == FREE_MUTEX_ID) allDone = true; \
                                        if (MUTEX == warpIdWAM){ /* We Won */

                                            // User code comes here

#define GIVE_WARP_MUTEX2(MUTEX)             doneWAM = true; \
                                            MUTEX = FREE_MUTEX_ID; \
                                        } \
                                    } \
                            } while(0)
#endif




#if USE_BALLOT_HISTOGRAM

template <typename OUTPUTTYPE>
static inline __device__
OUTPUTTYPE mySillyPopCount(unsigned int mymask, OUTPUTTYPE zero)
{
    return zero;
}

static inline __device__
int mySillyPopCount(unsigned int mymask, int zero)
{
    return (int)__popc(mymask);
}

static inline __device__
unsigned int mySillyPopCount(unsigned int mymask, unsigned int zero)
{
    return (unsigned int)__popc(mymask);
}

static inline __device__
long long mySillyPopCount(unsigned int mymask, long long zero)
{
    return (long long)__popc(mymask);
}

static inline __device__
unsigned long long mySillyPopCount(unsigned int mymask, unsigned  long long zero)
{
    return (unsigned long long)__popc(mymask);
}



template <histogram_type histotype, bool checkNSame, typename SUMFUNTYPE, typename OUTPUTTYPE>
static inline __device__
bool ballot_makeUnique(
    SUMFUNTYPE sumfunObj,
    int myKey, OUTPUTTYPE* myOut, OUTPUTTYPE* s_vals, int* s_keys, int* nSameKeys)
{

  unsigned int mymask;

/*  #if HBLOCK_SIZE != 32
  #error Please use threadblocks of 32 threads
  #endif*/
  //startKey = s_keys[startIndex];
  // First dig out for each thread who are the other threads that have the same key as us...
  //int i = 0;
  if (checkNSame) {
      unsigned int donemask = 0;
      int startIndex = 32 - 1;
      int startKey = s_keys[startIndex];
      *nSameKeys = 0;
      while (~donemask != 0 /*&& i++ < 32*/)
      {
        unsigned int mask = __ballot(myKey == startKey);
        if (myKey == startKey)
          mymask = mask;
        donemask |= mask;
        {
            int nSame = __popc(mask);
            if (nSame > *nSameKeys)
              *nSameKeys = nSame;
        }
        startIndex = 31 - __clz(~donemask);
        //if (myKey == 0)  printf("Startindex = %d, donemask = 0x%08x, mask = 0x%08x\n", startIndex, donemask, mask);
        if (startIndex >= 0)
          startKey = s_keys[startIndex];
      }
  } else {
      unsigned int donemask = 0;
      int startIndex = 32 - 1;
      while (startIndex >= 0)
      {
        int startKey = s_keys[startIndex];
        unsigned int mask = __ballot(myKey == startKey);
        if (myKey == startKey)
          mymask = mask;
        donemask |= mask;
        startIndex = 31 - __clz(~donemask);
      }
  }

  // Ok now mymask contains those threads - now we just reduce locally - all threads run at the same
  // time, but reducing threads lose always half of them with each iteration - it would help
  // to work with more than 32 entries, but the algorithm seems to get tricky there.
  {
    // Compute the left side of the mask and the right side. rmask first will contain our thread index, but
    // we zero it out immediately
    unsigned int lmask = (mymask >> (threadIdx.x & 31)) << (threadIdx.x & 31);
    int IamNth = __popc(lmask) - 1;
    bool Iwrite = IamNth == 0;
    if (histotype == histogram_atomic_inc)
    {
        // Fast-path for atomic inc
        *myOut = mySillyPopCount(mymask, *myOut);
        return Iwrite && (myKey >= 0);
    }
    else
    {
        unsigned int rmask = mymask & (~lmask);
        // Now compute which number is our thread in the subarray of those threads that have the same key
        // starting from the left (ie. index == 31). So for thread 31 this will be always zero.

        int nextIdx = 31 - __clz(rmask);

        s_vals[(threadIdx.x & 31)] = *myOut;
        //if (myKey == 0) printf("tid = %02d, IamNth = %02d, mask = 0x%08x, rmask = 0x%08x \n", threadIdx.x, IamNth, mymask, rmask);
        //bool done = __all(nextIdx < 0);
        // TODO: Unroll 5?
        while (!__all(nextIdx < 0))
        {
          // Reduce towards those threads that have lower IamNth
          // Our thread reads the next one if our internal ID is even
          if ((IamNth & 0x1) == 0)
          {
            if (nextIdx >= 0){
              // if (myKey == 0) printf("tid:%02d, add with %02d\n", threadIdx.x, nextIdx);
              *myOut = sumfunObj(*myOut, s_vals[nextIdx]);
            }
            // And writes to the shared memory if our internal ID is third on every 4-long subarray:
            if ((IamNth & 0x3) == 2)
            {
              // if (myKey == 0) printf("Tid %02d, store\n", threadIdx.x);
              s_vals[(threadIdx.x & 31)] = *myOut;
            }
          }
          // Now the beautiful part: Kill every other bit in the rmask bitfield. How, you ask?
          // Using ballot: Every bit we want to kill has IamNth odd, or conversely, we only
          // want to keep those bits that have IamNth even...
          rmask &= __ballot((IamNth & 0x1) == 0);
          nextIdx = 31 - __clz(rmask);
          // if (myKey == 0) printf("tid = %02d, next = %02d, key = %d\n", threadIdx.x, rmask, nextIdx, myKey);

          IamNth >>= 1;
          //printf("i = %d\n", i);
        }
        // And voila, we are done - write out the result:
        return Iwrite && (myKey >= 0);
    }
  }
}
#endif



template <bool laststeps, typename SUMFUNTYPE, typename OUTPUTTYPE>
static inline __device__
void myAtomicWarpAdd(OUTPUTTYPE* addr, OUTPUTTYPE val, volatile int* keyAddr, SUMFUNTYPE sumfunObj, bool Iwrite, int* warpmutex)
{
    // Taken from http://forums.nvidia.com/index.php?showtopic=72925
    // This is a tad slow, but allows arbitrary operation
    // For writes of 16 bytes or less AtomicCAS could be faster
    // (See CUDA programming guide)
    TAKE_WARP_MUTEX(0);
    //__shared__ int warpmutex;
    //INIT_WARP_MUTEX2(*warpmutex);
    //TAKE_WARP_MUTEX2(*warpmutex);
    bool write = Iwrite;
#define MU_TEMP_MAGIC 0xffffaaaa
    *keyAddr = MU_TEMP_MAGIC;
    while (1)
    {
        // Vote whose turn is it - remember, one thread does succeed always!:
        if (write) *keyAddr = threadIdx.x;
        if (*keyAddr == MU_TEMP_MAGIC)
            break;
        if (*keyAddr == threadIdx.x) // We won!
        {
            // Do arbitrary atomic op:
            *addr = sumfunObj(*addr, val);
            write = false;
            *keyAddr = MU_TEMP_MAGIC;
        }
    }
    GIVE_WARP_MUTEX(0);
    //GIVE_WARP_MUTEX2(*warpmutex);
#undef MU_TEMP_MAGIC
}



template <typename SUMFUNTYPE, typename OUTPUTTYPE>
static inline __device__
void myAtomicAdd(OUTPUTTYPE* addr, OUTPUTTYPE val, volatile int* keyAddr, SUMFUNTYPE sumfunObj)
{
    // Taken from http://forums.nvidia.com/index.php?showtopic=72925
    // This is a tad slow, but allows arbitrary operation
    // For writes of 16 bytes or less AtomicCAS could be faster
    // (See CUDA programming guide)
    bool write = true;
#define MU_TEMP_MAGIC 0xffffaaaa
    *keyAddr = MU_TEMP_MAGIC;
    while (1)
    {
        // Vote whose turn is it - remember, one thread does succeed always!:
        if (write ) *keyAddr = threadIdx.x;
        if (*keyAddr == MU_TEMP_MAGIC)
            break;
        if (*keyAddr == threadIdx.x) // We won!
        {
            // Do arbitrary atomic op:
            *addr = sumfunObj(*addr, val);
            write = false;
            *keyAddr = MU_TEMP_MAGIC;
        }
    }
#undef MU_TEMP_MAGIC
}

/*static __inline__ __device__ unsigned long long int atomicAdd(unsigned long long int *address, unsigned long long int val)
{
  return __ullAtomicAdd(address, val);
}*/


template <typename OUTPUTTYPE>
static inline __device__
void atomicAdd(OUTPUTTYPE* addr, OUTPUTTYPE val)
{
    //*addr = val;
}


template <typename OUTPUTTYPE>
static inline __device__
void atomicAdd(OUTPUTTYPE* addr, int val)
{
    //*addr = val;
}

#if 0
template <typename OUTPUTTYPE>
static inline __device__
void atomicAdd(OUTPUTTYPE* addr, float val)
{
    //*addr = val;
}
#endif

template <typename OUTPUTTYPE>
static inline __device__
void atomicAdd(OUTPUTTYPE* addr, unsigned int val)
{
    //*addr = val;
}



template <typename SUMFUNTYPE, typename OUTPUTTYPE>
static inline __device__
void myAtomicAddStats(OUTPUTTYPE* addr, OUTPUTTYPE val, volatile int* keyAddr, SUMFUNTYPE sumfunObj, int* nSameOut, bool Iwrite)
{
    // Taken from http://forums.nvidia.com/index.php?showtopic=72925
    bool write = true;
    *keyAddr = 0xffffffff;
    while (Iwrite)
    {
        // Vote whose turn is it - remember, one thread does succeed always!:
        if (write ) *keyAddr = threadIdx.x;
        if (*keyAddr == 0xffffffff)
            break;
        if (*keyAddr == threadIdx.x) // We won!
        {
            // Do arbitrary atomic op:
            *addr = sumfunObj(*addr, val);
            write = false;
            *keyAddr = 0xffffffff;
        } else {
            *nSameOut = *nSameOut + 1;
        }
    }

    {
        // Then find max
        __shared__ int nSame[HBLOCK_SIZE];
        nSame[threadIdx.x] = *nSameOut;
#define TMPMAX(A,B) (A) > (B) ? (A) : (B)
#define tidx threadIdx.x
        if (tidx < 16) nSame[tidx] = TMPMAX(nSame[tidx] , nSame[tidx + 16]);
        if (tidx < 8) nSame[tidx] = TMPMAX(nSame[tidx] , nSame[tidx + 8]);
        if (tidx < 4) nSame[tidx] = TMPMAX(nSame[tidx] , nSame[tidx + 4]);
        if (tidx < 2) nSame[tidx] = TMPMAX(nSame[tidx] , nSame[tidx + 2]);
        if (tidx < 1) nSame[tidx] = TMPMAX(nSame[tidx] , nSame[tidx + 1]);
#undef TMPMAX
#undef tidx
        // Broadcast to all threads
        *nSameOut = nSame[0];
    }
}
// TODO: Make unique within one warp?
template<histogram_type histotype, bool checkNSame, typename SUMFUNTYPE, typename OUTPUTTYPE>
static inline __device__
bool reduceToUnique(OUTPUTTYPE* res, int myKey, int* nSame, SUMFUNTYPE sumfunObj, int* keys, OUTPUTTYPE* outputs)
{
    keys[(threadIdx.x & 31)] = myKey;
#if  USE_BALLOT_HISTOGRAM
    return ballot_makeUnique<histotype, checkNSame>(sumfunObj, myKey, res, outputs, keys, nSame);
#else
    {

      int i;
      bool writeResult = myKey >= 0;
      int myIdx = (threadIdx.x & 31) + 1;
      outputs[(threadIdx.x & 31)] = *res;
      // The assumption for sanity of this loop here is that all the data is in registers or shared memory and
      // hence this loop will not actually be __that__ slow.. Also it helps if the data is spread out (ie. there are
      // a lot of different indices here)
      for (i = 1; i < 32 && writeResult; i++)
      {
        if (myIdx >= 32)
          myIdx = 0;
        // Is my index the same as the index on the index-list?
        if (keys[myIdx] == myKey /*&& threadIdx.x != myIdx*/)
        {
          if (checkNSame) (*nSame)++;
          // If yes, then we can sum up the result using users sum-functor
          *res = sumfunObj(*res, outputs[myIdx]);
          // But if somebody else is summing up this index already, we don't need to (wasted effort done here)
          if (myIdx < threadIdx.x)
            writeResult = false;
        }
        myIdx++;
      }
      // Ok - we are done - now we can proceed in writing the result (if some other thread isn't doing it already)
      if (checkNSame)
      {
          // Manual reduce
          int tid = threadIdx.x;
          keys[tid] = *nSame;
          if (tid < 16) keys[tid] = keys[tid] > keys[tid + 16] ? keys[tid] : keys[tid+16];
          if (tid < 8) keys[tid] = keys[tid] > keys[tid + 8] ? keys[tid] : keys[tid+8];
          if (tid < 4) keys[tid] = keys[tid] > keys[tid + 4] ? keys[tid] : keys[tid+4];
          if (tid < 2) keys[tid] = keys[tid] > keys[tid + 2] ? keys[tid] : keys[tid+2];
          if (tid < 1) keys[tid] = keys[tid] > keys[tid + 1] ? keys[tid] : keys[tid+1];
          *nSame = keys[0];
      }
      return writeResult;
    }
#endif
}


static inline __host__ __device__
void checkStrategyFun(bool *reduce, int nSame, int nSameTot, int step, int nBinSetslog2)
{
#if __CUDA_ARCH__ >= 200
#define STR_LIMIT 12
#else
#define STR_LIMIT 24
#endif
// TODO: Fix average case - a lot of things to tune here...
    if ((nSameTot > STR_LIMIT * step || nSame > STR_LIMIT))
        *reduce = true;
    else
        *reduce = false;
#undef STR_LIMIT
}

// Special case for floats (atomicAdd works only from __CUDA_ARCH__ 200 and up)
template <typename SUMFUNTYPE>
static inline __device__
void wrapAtomicAdd2(float* addr, float val, int* key, SUMFUNTYPE sumFunObj)
{
    //*addr = val;
#if __CUDA_ARCH__ >= 200
  atomicAdd(addr, val);
#else
  myAtomicAdd(addr, val, key, sumFunObj);
#endif
}

template <typename SUMFUNTYPE,typename OUTPUTTYPE>
static inline __device__
void wrapAtomicAdd2(OUTPUTTYPE* addr, OUTPUTTYPE val, int* key, SUMFUNTYPE sumFunObj)
{
  atomicAdd(addr, val);
}


// Special case for floats (atomicAdd works only from __CUDA_ARCH__ 200 and up)
template <bool laststeps, typename SUMFUNTYPE>
static inline __device__
void wrapAtomicAdd2Warp(float* addr, float val, int* key, SUMFUNTYPE sumFunObj, bool Iwrite, int* warpmutex)
{
    //*addr = val;
#if __CUDA_ARCH__ >= 200
  if (Iwrite) atomicAdd(addr, val);
#else
  myAtomicWarpAdd<laststeps>(addr, val, key, sumFunObj, Iwrite, warpmutex);
#endif
}

template <bool laststeps, typename SUMFUNTYPE,typename OUTPUTTYPE>
static inline __device__
void wrapAtomicAdd2Warp(OUTPUTTYPE* addr, OUTPUTTYPE val, int* key, SUMFUNTYPE sumFunObj, bool Iwrite, int* warpmutex)
{
  if (Iwrite) atomicAdd(addr, val);
}






template <typename OUTPUTTYPE, typename SUMFUNTYPE>
static inline __device__
void wrapAtomicAdd(OUTPUTTYPE* addr, OUTPUTTYPE val, int* key, SUMFUNTYPE sumFunObj)
{
    //*addr = val;
#if __CUDA_ARCH__ >= 120
  wrapAtomicAdd2(addr, val, key, sumFunObj);
#else
  myAtomicAdd(addr, val, key, sumFunObj);
#endif
}
template <typename OUTPUTTYPE, typename SUMFUNTYPE>
static inline __device__
void wrapAtomicInc(OUTPUTTYPE* addr, int* key, SUMFUNTYPE sumFunObj)
{
    //*addr = val;
#if __CUDA_ARCH__ >= 120
  wrapAtomicAdd2((int*)addr, 1, key, sumFunObj);
#else
  //myAtomicAdd((int*)addr, 1, key, sumFunObj);
#endif
}

template <typename SUMFUNTYPE>
static inline __device__
void wrapAtomicInc(int* addr, int* key, SUMFUNTYPE sumFunObj)
{
    //*addr = val;
#if __CUDA_ARCH__ >= 120
  wrapAtomicAdd2(addr, 1, key, sumFunObj);
#else
  myAtomicAdd(addr, 1, key, sumFunObj);
#endif
}






template <bool laststeps, typename OUTPUTTYPE, typename SUMFUNTYPE>
static inline __device__
void wrapAtomicAddWarp(OUTPUTTYPE* addr, OUTPUTTYPE val, int* key, SUMFUNTYPE sumFunObj, bool Iwrite, int* warpmutex)
{
    //*addr = val;
#if __CUDA_ARCH__ >= 120
  wrapAtomicAdd2Warp<laststeps>(addr, val, key, sumFunObj, Iwrite, warpmutex);
#else
  myAtomicWarpAdd<laststeps>(addr, val, key, sumFunObj, Iwrite, warpmutex);
#endif
}
template <bool laststeps, typename OUTPUTTYPE, typename SUMFUNTYPE>
static inline __device__
void wrapAtomicIncWarp(OUTPUTTYPE* addr, int* key, SUMFUNTYPE sumFunObj, bool Iwrite, int* warpmutex)
{
    //*addr = val;
#if __CUDA_ARCH__ >= 120
  wrapAtomicAdd2Warp<laststeps>((int*)addr, 1, key, sumFunObj, Iwrite, warpmutex);
#else
  //myAtomicAdd((int*)addr, 1, key, sumFunObj);
#endif
}

template <bool laststeps, typename SUMFUNTYPE>
static inline __device__
void wrapAtomicIncWarp(int* addr, int* key, SUMFUNTYPE sumFunObj, bool Iwrite, int* warpmutex)
{
    //*addr = val;
#if __CUDA_ARCH__ >= 120
  wrapAtomicAdd2Warp<laststeps>(addr, 1, key, sumFunObj, Iwrite, warpmutex);
#else
  myAtomicWarpAdd<laststeps>(addr, 1, key, sumFunObj, Iwrite, warpmutex);
#endif
}








// TODO: Consider the following:
// First private hash for each warp - later, share hash-tables between warps
// Try also: private hashes for some threads of one warp etc

template <typename OUTPUTTYPE>
struct myHash
{
    int* keys;
#if !USE_ATOMICS_HASH
    int* locks;
#endif
    OUTPUTTYPE* vals;
    OUTPUTTYPE* myBlockOut;
};



template <typename OUTPUTTYPE>
static inline __device__
void InitHash(struct myHash<OUTPUTTYPE> *hash, OUTPUTTYPE zero, int hashSizelog2)
{
    int nloops = (1 << hashSizelog2) >> LBLOCK_SIZE_LOG2;
    int* myEntry = &hash->keys[threadIdx.x];
    for (int i = 0; i < nloops; i++)
    {
        *myEntry = -1;
        myEntry += LBLOCK_SIZE;
    }
    if ((nloops << LBLOCK_SIZE_LOG2) + threadIdx.x < (1 << hashSizelog2))
    {
        *myEntry = -1;
    }
    // Done
}

#if 0 // OLD code
template <typename SUMFUNTYPE, typename OUTPUTTYPE>
static inline __device__
void FlushHash(struct myHash<OUTPUTTYPE> *hash, SUMFUNTYPE sumfunObj, int hashSizelog2)
{
    int nloops = (1 << hashSizelog2) >> LBLOCK_SIZE_LOG2;
    OUTPUTTYPE* myVal = &hash->vals[threadIdx.x];
    int* key = &hash->keys[threadIdx.x];
    for (int i = 0; i < nloops; i++) {
        int keyIndex = *key;
        if (keyIndex >= 0) {
            hash->myBlockOut[keyIndex] = sumfunObj(*myVal, hash->myBlockOut[keyIndex]);
            *key = -1;
        }
        key += LBLOCK_SIZE;
        myVal += LBLOCK_SIZE;
    }
    if ((nloops << LBLOCK_SIZE_LOG2) + threadIdx.x < (1 << hashSizelog2))
    {
      int keyIndex = *key;
      if (keyIndex >= 0){
          hash->myBlockOut[keyIndex] = sumfunObj(*myVal, hash->myBlockOut[keyIndex]);
          *key = -1;
      }
    }
}
#endif // 0

// See: http://www.burtleburtle.net/bob/hash/doobs.html
// Mix by Bob Jenkins
#define HISTO_JENKINS_MIX(A, B, C)  \
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

static inline __device__
unsigned int histogramHashFunction(int key)
{
#if USE_JENKINS_HASH
    unsigned int a = (unsigned int)key;
    unsigned int c,b;
    // TODO: What are good constants?
    b = 0x9e3779b9;
    c = 0xf1232345;
    HISTO_JENKINS_MIX(a, b, c);
    return c;
#else
    // Golden ratio hash
    return (0x9e3779b9u * (unsigned int)key);
#endif
}


#if USE_ATOMICS_HASH
template <typename SUMFUNTYPE, typename OUTPUTTYPE>
static inline __device__
void AddToHash(OUTPUTTYPE res, int myKey, struct myHash<OUTPUTTYPE> *hash, SUMFUNTYPE sumfunObj, int hashSizelog2, bool Iwrite, bool unique)
{
    if (unique)
    {
        if (Iwrite)
        {
            hash->myBlockOut[myKey] = sumfunObj(res, hash->myBlockOut[myKey]);
        }
        return;
    }
    unsigned int hashkey = histogramHashFunction(myKey);
    volatile __shared__ bool hashFull;
    int index = (int)(hashkey >> (32 - hashSizelog2));
    bool Iamdone = !Iwrite;
    bool IFlush = Iwrite;

    hashFull = true;
    while (hashFull)
    {
        // Mark here hash full, and if any thread has problems finding
        // free entry in hash, then that thread sets hashFull to nonzero
        if (threadIdx.x == 0) hashFull = false;
        // Do atomic-part
        int old = -2;
        int expect = -1;
        while (!Iamdone && !hashFull)
        {
            old = atomicCAS(&hash->keys[index], expect, -3);
            if (old == expect) // We won!
            {
                int key = old;
                if (key == -1 || key == myKey)
                {
                    if (key == -1)
                    {
                        hash->vals[index] = res;
                    }
                    else
                    {
                        hash->vals[index] = sumfunObj(res, hash->vals[index]);
                        IFlush = false;
                    }
                    hash->keys[index] = myKey;
                    Iamdone = true;
                }
                else
                {
                    hashFull = true;
                    hash->keys[index] = key;
                    expect = -1;
                }
            }
            else
            {

                if (old != myKey)
                {
                    hashFull = true;
                    expect = -1;
                }
                else
                {
                    expect = old;
                }
            }
        }
        if (IFlush && Iamdone)
        {
            OUTPUTTYPE* myVal = &hash->vals[index];
            int* key = &hash->keys[index];
            // TODO: Workaround - get rid of if. Where do the extra flushes come from?
            if (*key >= 0) hash->myBlockOut[*key] = sumfunObj(*myVal, hash->myBlockOut[*key]);
            //hash->myBlockOut[myKey] = sumfunObj(*myVal, hash->myBlockOut[myKey]);
            *key = -1;
        }
    }
}

#else
template <typename SUMFUNTYPE, typename OUTPUTTYPE>
static inline __device__
void AddToHash(OUTPUTTYPE res, int myKey, struct myHash<OUTPUTTYPE> *hash, SUMFUNTYPE sumfunObj, int hashSizelog2, bool Iwrite, bool unique)
{
    if (unique)
    {
        if (Iwrite)
        {
            hash->myBlockOut[myKey] = sumfunObj(res, hash->myBlockOut[myKey]);
        }
        return;
    }
    unsigned int hashkey = histogramHashFunction(myKey);
    volatile __shared__ int hashFull;
    int index = (int)(hashkey >> (32 - hashSizelog2));
    bool Iamdone = false;
    bool IFlush = Iwrite;

    // TODO: syncthreads()...
    hashFull = -10;
    while (hashFull != 0)
    {
        volatile int* lock = &hash->locks[index];
        bool write = Iwrite;
#define TMP_LOCK_MAGIC  0xfffffffe
        *lock = TMP_LOCK_MAGIC;
        // Mark here hash full, and if any thread has problems finding
        // free entry in hash, then that thread sets hashFull to nonzero
        if (threadIdx.x == 0) hashFull = 0;
        // Do atomic-part
        while (1)
        {
            if (!Iamdone && write) *lock = threadIdx.x;
            if (*lock == TMP_LOCK_MAGIC)
                break;
            if (*lock == threadIdx.x) // We won!
            {
                int key = hash->keys[index];
                if (key == -1)
                {
                    hash->keys[index] = myKey;
                    hash->vals[index] = res;
                    Iamdone = true;
                }
                else if (key == myKey)
                {
                    hash->vals[index] = sumfunObj(res, hash->vals[index]);
                    Iamdone = true;
                    IFlush = false;
                }
                else
                {
                    hashFull = 1;
                }
                // Do arbitrary atomic op:
                write = false;
                *lock = TMP_LOCK_MAGIC;
            }
        }
        if (IFlush)
        {
            OUTPUTTYPE* myVal = &hash->vals[index];
            int* key = &hash->keys[index];
            // TODO: Workaround - get rid of if. Where do the extra flushes come from?
            if (*key >= 0) hash->myBlockOut[*key] = sumfunObj(*myVal, hash->myBlockOut[*key]);
            *key = -1;
        }
    }
#undef TMP_LOCK_MAGIC
}
#endif


template <histogram_type histotype, int nMultires, bool reduce, bool checkStrategy, bool laststep, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
static inline __device__
void histo_largenbin_step(INPUTTYPE input, TRANSFORMFUNTYPE xformObj, SUMFUNTYPE sumfunObj, OUTPUTTYPE zero,
                INDEXT* myStart, INDEXT end, struct myHash<OUTPUTTYPE> *hash, OUTPUTTYPE* blockOut, int nOut, int stepNum, int stepsleft, int* nSameTot, bool* reduceOut, int hashSizelog2,
                OUTPUTTYPE* rOuts, int* rKeys)
{
    if (!laststep)
    {
        if (checkStrategy)
        {
            int myKeys[nMultires];
            int nSame = 0;
            OUTPUTTYPE res[nMultires];
            xformObj(input, *myStart, &myKeys[0], &res[0], nMultires);
            // TODO: Unroll? addtoHash is a big function.. Hmm but, unrolling would enable registers probably
            bool Iwrite;
#define     ADD_ONE_RESULT(RESIDX, NSAME, CHECK)                                                                               	\
            do { if (RESIDX < nMultires) {                                                                                     	\
                Iwrite = reduceToUnique<histotype, CHECK>                                                                      	\
                    (&res[RESIDX % nMultires], myKeys[RESIDX % nMultires], NSAME, sumfunObj, rKeys, rOuts);             		\
                if ((threadIdx.x) < (1 << hashSizelog2)) hash->keys[threadIdx.x] = -1;                                          \
                AddToHash(res[RESIDX % nMultires], myKeys[RESIDX % nMultires], hash, sumfunObj, hashSizelog2, Iwrite, true);  	\
            } } while (0)
            ADD_ONE_RESULT(0, &nSame, true);
            ADD_ONE_RESULT(1, NULL, false);
            ADD_ONE_RESULT(2, NULL, false);
            ADD_ONE_RESULT(3, NULL, false);
#undef ADD_ONE_RESULT
            //#pragma unroll
            for (int resid = 4; resid < nMultires; resid++)
            {
                bool Iwrite = reduceToUnique<histotype, false>(&res[resid], myKeys[resid], NULL, sumfunObj, rKeys, rOuts);
                if ((threadIdx.x) < (1 << hashSizelog2)) hash->keys[threadIdx.x] = -1;
                AddToHash(res[resid], myKeys[resid], hash, sumfunObj, hashSizelog2, Iwrite, true);
            }
            *nSameTot += nSame;
            checkStrategyFun(reduceOut, nSame, *nSameTot, stepNum, 0);
            *myStart += LBLOCK_SIZE;
        }
        else
        {
            INDEXT startLim = *myStart + ((LBLOCK_SIZE << LARGE_NBIN_CHECK_INTERVAL_LOG2) - LBLOCK_SIZE);
            for (; *myStart < startLim; *myStart += LBLOCK_SIZE)
            {
                int myKeys[nMultires];
                OUTPUTTYPE res[nMultires];
                xformObj(input, *myStart, &myKeys[0], &res[0], nMultires);
                //#pragma unroll
                bool Iwrite = true;
#define ADD_ONE_RESULT(RES) \
                do { if (RES < nMultires) { \
                  if (reduce){ Iwrite = reduceToUnique<histotype, false>(&res[RES % nMultires],         \
                                        myKeys[RES % nMultires], NULL, sumfunObj, rKeys, rOuts);        \
                               if (threadIdx.x < (1 << hashSizelog2)) hash->keys[threadIdx.x] = -1;}    \
                  AddToHash(res[RES % nMultires], myKeys[RES % nMultires], hash,                        \
                                sumfunObj, hashSizelog2, Iwrite, reduce);                               \
                 } } while (0)
                ADD_ONE_RESULT(0);
                ADD_ONE_RESULT(1);
                ADD_ONE_RESULT(2);
                ADD_ONE_RESULT(3);
#undef ADD_ONE_RESULT
                for (int resid = 4; resid < nMultires; resid++)
                {
                    bool Iwrite = true;
                    if (reduce){
                        Iwrite = reduceToUnique<histotype, false>(&res[resid], myKeys[resid], NULL, sumfunObj, rKeys, rOuts);
                        if (threadIdx.x < (1 << hashSizelog2)) hash->keys[threadIdx.x] = -1;
                    }
                    AddToHash(res[resid], myKeys[resid], hash, sumfunObj, hashSizelog2, Iwrite, reduce);
                }
            }
        }
    }
    else // These are the last steps then
    {
        for (int substep = 0; substep < stepsleft; substep++)
        {

            int myKeys[nMultires];
            OUTPUTTYPE res[nMultires];
            bool Iwrite = false;
            if (*myStart < end)
            {
                Iwrite = true;
                xformObj(input, *myStart, &myKeys[0], &res[0], nMultires);
            }
            else
            {
            #pragma unroll
              for (int resid = 0; resid < nMultires; resid++)
              {
                res[resid] = zero;
                myKeys[resid] = 0;
              }
            }
            //#pragma unroll
            {
                bool Iwrite2 = Iwrite;
#define     ADD_ONE_RESULT(RES)             																				\
                do { if (RES < nMultires) { 																				\
                  if (reduce){ Iwrite2 = reduceToUnique<histotype, false>                                          			\
                    (&res[RES % nMultires], myKeys[RES % nMultires], NULL, sumfunObj, rKeys, rOuts);            			\
                    if (threadIdx.x < (1 << hashSizelog2)) hash->keys[threadIdx.x] = -1; }                                  \
                  AddToHash(res[RES % nMultires], myKeys[RES % nMultires], hash, sumfunObj, hashSizelog2, Iwrite2, reduce);	\
                } } while(0)

                ADD_ONE_RESULT(0);
                ADD_ONE_RESULT(1);
                ADD_ONE_RESULT(2);
                ADD_ONE_RESULT(3);
                #undef ADD_ONE_RESULT
                for (int resid = 4; resid < nMultires; resid++)
                {
                    //bool Iwrite2 = true;
                    if (reduce){
                        Iwrite2 = reduceToUnique<histotype, false>(&res[resid], myKeys[resid], NULL, sumfunObj, rKeys, rOuts);
                        if (threadIdx.x < (1 << hashSizelog2)) hash->keys[threadIdx.x] = -1;
                    }
                    AddToHash(res[resid], myKeys[resid], hash, sumfunObj, hashSizelog2, Iwrite2, reduce);
                }
            }
            *myStart += LBLOCK_SIZE;
        }
    }
}


template <histogram_type histotype, int nMultires, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
__global__
void histo_kernel_largeNBins(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero,
    OUTPUTTYPE* blockOut, int nOut,
    int nSteps,
    int hashSizelog2)
{
    extern __shared__ int keys[];
#if USE_ATOMICS_HASH
    OUTPUTTYPE* vals = (OUTPUTTYPE*)(&keys[1 << hashSizelog2]);
    if (hashSizelog2 < LBLOCK_SIZE_LOG2)
        vals = &keys[1 << LBLOCK_SIZE_LOG2];
#else
    int* locks = &keys[1 << hashSizelog2];
    if (hashSizelog2 < LBLOCK_SIZE_LOG2)
        locks = &keys[1 << LBLOCK_SIZE_LOG2];
    OUTPUTTYPE* vals = (OUTPUTTYPE*)(&locks[1 << hashSizelog2]);
#endif
    /*int* rKeys = (int*)(&vals[1 << hashSizelog2]);
    OUTPUTTYPE* rOuts = (OUTPUTTYPE*)(&rKeys[LBLOCK_SIZE]);*/

    int* rKeys = &keys[0];
    OUTPUTTYPE* rOuts = vals;

    struct myHash<OUTPUTTYPE> hash;

    hash.keys = keys;
#if !USE_ATOMICS_HASH
    hash.locks = locks;
#endif
    hash.vals = vals;
    // Where do we put the results from our warp (block)?
    hash.myBlockOut = &blockOut[nOut * blockIdx.x];

    INDEXT myStart = start + (INDEXT)(((blockIdx.x * nSteps) << LBLOCK_SIZE_LOG2) + threadIdx.x);
    // Assert that myStart is not out of bounds!
    int nFullSteps = nSteps >> LARGE_NBIN_CHECK_INTERVAL_LOG2;
    bool reduce = false;


    InitHash(&hash, zero, hashSizelog2);
    int nSameTot = 0;
    for (int fstep = 0; fstep < nFullSteps; fstep++)
    {
        int stepNum = fstep << LARGE_NBIN_CHECK_INTERVAL_LOG2;
        histo_largenbin_step<histotype, nMultires, true, true, false,INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
            (input, xformObj, sumfunObj, zero, &myStart, end, &hash, blockOut, nOut, stepNum, 0, &nSameTot, &reduce, hashSizelog2, rOuts, rKeys);
        if (reduce) {
            histo_largenbin_step<histotype, nMultires, true, false, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
                (input, xformObj, sumfunObj, zero, &myStart, end, &hash, blockOut, nOut, stepNum + 1, 0, &nSameTot, &reduce, hashSizelog2, rOuts, rKeys);
        } else {
            histo_largenbin_step<histotype, nMultires, false, false, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
                (input, xformObj, sumfunObj, zero, &myStart, end, &hash, blockOut, nOut, stepNum + 1, 0, &nSameTot, &reduce, hashSizelog2, rOuts, rKeys);
        }
    }
    // Last steps
    int nstepsleft = nSteps - (nFullSteps << LARGE_NBIN_CHECK_INTERVAL_LOG2);
    if (nstepsleft > 0)
    {
        int stepNum = nFullSteps << LARGE_NBIN_CHECK_INTERVAL_LOG2;
        if (reduce)
            histo_largenbin_step<histotype, nMultires, true, false, true, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
                (input, xformObj, sumfunObj, zero, &myStart, end, &hash, blockOut, nOut, stepNum, nstepsleft, &nSameTot, &reduce, hashSizelog2, rOuts, rKeys);
        else
            histo_largenbin_step<histotype, nMultires, false, false, true, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
                (input, xformObj, sumfunObj, zero, &myStart, end, &hash, blockOut, nOut, stepNum, nstepsleft, &nSameTot, &reduce, hashSizelog2, rOuts, rKeys);
    }
    // Flush values still in hash
    //FlushHash(&hash, sumfunObj, hashSizelog2);
}

#if USE_MEDIUM_PATH

//

template <histogram_type histotype, int nMultires, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
__global__
void histo_kernel_mediumNBins(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero,
    OUTPUTTYPE* blockOut, int nOut,
    int nSteps)
{
#if __CUDA_ARCH__ >= 120
    OUTPUTTYPE* ourOut = &blockOut[nOut * (threadIdx.x % MED_THREAD_DEGEN) * blockIdx.x];
    INDEXT myStart = start + (INDEXT)(((blockIdx.x * nSteps) << MEDIUM_BLOCK_SIZE_LOG2) + threadIdx.x);
    bool reduce = false;
    int nSameTot = 0;
    for (int step = 0; step < nSteps - 1; step++)
    {
        bool check = false;
        int myKey[nMultires];
        OUTPUTTYPE myOut[nMultires];
        xformObj(input, myStart, &myKey[0], &myOut[0],nMultires);
        // TODO: magic constant
        if ((step & 63) == 0)
            check = true;
        {
            int nSame;
            __shared__ int keys[MEDIUM_BLOCK_SIZE];
            __shared__ OUTPUTTYPE rOut[MEDIUM_BLOCK_SIZE];
            int warpIdx = threadIdx.x >> 5;
            int* wkeys = &keys[warpIdx << 5];
            OUTPUTTYPE* wOut = &rOut[warpIdx << 5];
            bool Iwrite;
#define ADD_ONE_RESULT(RESID)                                                                   \
            do { if (RESID < nMultires) {                                                       \
                if (reduce || check){                                                           \
                    if (check) Iwrite = reduceToUnique<histotype, true>                         \
                        (&myOut[RESID % nMultires], myKey[RESID % nMultires],                   \
                            &nSame, sumfunObj, wkeys, wOut);                                    \
                    else Iwrite = reduceToUnique<histotype, false>                              \
                        (&myOut[RESID % nMultires], myKey[RESID % nMultires], NULL, sumfunObj,  \
                            wkeys, wOut);                                                       \
                    if (Iwrite)                                                                 \
                        atomicAdd(&ourOut[myKey[RESID % nMultires]], myOut[RESID % nMultires]); \
                    if (check){                                                                 \
                        nSameTot += nSame;                                                      \
                        checkStrategyFun(&reduce, nSame, nSameTot, step, 0);                    \
                        check = false;                                                          \
                    }                                                                           \
                } else {                                                                        \
                    if (histotype == histogram_atomic_inc)                                      \
                        atomicAdd(&ourOut[myKey[RESID % nMultires]], 1);                        \
                    else if (histotype == histogram_atomic_add)                                 \
                        atomicAdd(&ourOut[myKey[RESID % nMultires]], myOut[RESID % nMultires]); \
                } }                                                                             \
            } while(0)
            ADD_ONE_RESULT(0);
            ADD_ONE_RESULT(1);
            ADD_ONE_RESULT(2);
            ADD_ONE_RESULT(3);
 //#pragma unroll
            for (int resid = 4; resid < nMultires; resid++)
            {
                ADD_ONE_RESULT(resid);
            }
        }
        myStart += MEDIUM_BLOCK_SIZE;
    }
    if (myStart < end)
    {
        int myKey[nMultires];
        OUTPUTTYPE myOut[nMultires];
        xformObj(input, myStart, &myKey[0], &myOut[0],nMultires);
        for (int resid = 0; resid < nMultires; resid++)
        {
            if (histotype == histogram_atomic_inc)
            {
                atomicAdd(&ourOut[myKey[resid]], 1);
            }
            else if (histotype == histogram_atomic_add)
            {
                atomicAdd(&ourOut[myKey[resid]], myOut[resid]);
            }
        }
    }
#endif // __CUDA_ARCH__
}
#endif // USE_MEDIUM_PATH





static int determineHashSizeLog2(size_t outSize, int* nblocks, cudaDeviceProp* props)
{
    // TODO: Magic hat-constant 500 reserved for inputs, how to compute?
    int sharedTot = (props->sharedMemPerBlock - 500) /* / LBLOCK_WARPS*/;
    //int sharedTot = 32000;
    // How many blocks of 32 keys could we have?
    //int nb32Max = sharedTot / (32 * outSize);
    // But ideally we should run at least 4 active blocks per SM,
    // How can we balance this? Well - with very low ablock-values (a),
	// we perform bad, but after 4, adding more
    // will help less and less, whereas adding more to the hash always helps!
#if USE_ATOMICS_HASH
    outSize += sizeof(int);
#else
    outSize += sizeof(int);
#endif
    int naMax = sharedTot / (32 * outSize);
    while (naMax > numActiveUpperLimit) naMax >>= 1;
    int nb32 = sharedTot / (32 * outSize * naMax);

    // Now we have "number of pieces", use it to compute some nice power-of-two hash-size
    int hashSize = nb32 * 32;
    unsigned int res = 0;
    if (hashSize >= 1<<16) { hashSize >>= 16; res += 16; }
    if (hashSize >= 1<< 8) { hashSize >>=  8; res +=  8; }
    if (hashSize >= 1<< 4) { hashSize >>=  4; res +=  4; }
    if (hashSize >= 1<< 2) { hashSize >>=  2; res +=  2; }
    if (hashSize >= 1<< 1) {                  res +=  1; }

    // Now res holds the log2 of hash size => n active blocksMEDIUM_BLOCK_SIZE_LOG2 = sharedTot / (outSize << res);
    *nblocks = (sharedTot / (outSize << res)) * props->multiProcessorCount;
    if (*nblocks > props->multiProcessorCount * 8) *nblocks = props->multiProcessorCount * 8;
    return res;
}


template <typename OUTPUTTYPE>
__global__
void initKernel(OUTPUTTYPE* tmpOut, OUTPUTTYPE zeroVal, int tmpOutSize, int steps)
{
  int idx = blockIdx.x * blockDim.x * steps + threadIdx.x;
  for (int step = 0; step < steps; step++)
  {
    if (idx < tmpOutSize)
      tmpOut[idx] = zeroVal;
    idx += blockDim.x;
  }
}

template <histogram_type histotype, typename OUTPUTTYPE>
static int getLargeBinTmpbufsize(int nOut, cudaDeviceProp* props, int cuda_arch)
{
    int nblocks;
    int hashSizelog2 = determineHashSizeLog2(sizeof(OUTPUTTYPE), &nblocks, props);
    int arrLen = nblocks;
#if USE_MEDIUM_PATH
        if (cuda_arch >= 120 && (histotype == histogram_atomic_inc || histotype == histogram_atomic_add))
            arrLen *= MED_THREAD_DEGEN;
#endif
    return (arrLen + 1) * nOut * sizeof(OUTPUTTYPE);
}

template <histogram_type histotype, int nMultires, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
static
void callHistogramKernelLargeNBins(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero, OUTPUTTYPE* out, int nOut,
    cudaDeviceProp* props, int cuda_arch, cudaStream_t stream,
    int* getTmpBufSize,
    void* tmpBuffer,
    bool outInDev)
{
    int nblocks;
    int hashSizelog2 = determineHashSizeLog2(sizeof(OUTPUTTYPE), &nblocks, props);
    INDEXT size = end - start;
    // Check if there is something to do actually...
    if (end <= start)
    {
      if (getTmpBufSize) getTmpBufSize = 0;
        return;
    }
    dim3 block = LBLOCK_SIZE;
    dim3 grid = nblocks;
    int arrLen = nblocks;
#if USE_MEDIUM_PATH
        if (cuda_arch >= 120 && (histotype == histogram_atomic_inc || histotype == histogram_atomic_add))
            arrLen *= MED_THREAD_DEGEN;
#endif
    INDEXT nSteps = size / (INDEXT)( LBLOCK_SIZE * nblocks);
    OUTPUTTYPE* tmpOut;
    //int n = nblocks;
    if (getTmpBufSize) {
        *getTmpBufSize = (arrLen + 1) * nOut * sizeof(OUTPUTTYPE);
        return;
    }

    if (tmpBuffer){
        tmpOut = (OUTPUTTYPE*)tmpBuffer;
    }
    else {
        size_t allocSize = (arrLen + 1) * nOut * sizeof(OUTPUTTYPE);
        cudaMalloc((void**)&tmpOut, allocSize);
    }

    //printf("Using hash-based histogram: hashsize = %d, nblocksToT = %d\n", (1 << hashSizelog2), nblocks);

#if USE_ATOMICS_HASH
    int extSharedNeeded = (1 << hashSizelog2) * (sizeof(OUTPUTTYPE) + sizeof(int));
#else
    int extSharedNeeded = (1 << hashSizelog2) * (sizeof(OUTPUTTYPE) + sizeof(int) * 2);
#endif

    // The shared memory here is needed for the reduction code (ie. reduce to unique)
    // TODO: new hash-code could probably reuse the memory reserved for the hash-table,
    // it would just need to reinit the keys to -1 after use - think about it.
    if (cuda_arch >= 200 && histotype == histogram_atomic_inc)
    {
        if (hashSizelog2 < LBLOCK_SIZE_LOG2)
            extSharedNeeded += (sizeof(int) << (LBLOCK_SIZE_LOG2 - hashSizelog2));
    }
    else
    {
        if (hashSizelog2 < LBLOCK_SIZE_LOG2)
            extSharedNeeded += ((sizeof(OUTPUTTYPE) + sizeof(int)) << (LBLOCK_SIZE_LOG2 - hashSizelog2));
    }
    //printf("binsets = %d, steps = %d\n", (1 << nKeysetslog2), nsteps);


    {
  #define IBLOCK_SIZE_LOG2    7
  #define IBLOCK_SIZE         (1 << IBLOCK_SIZE_LOG2)
      int initPaddedSize =
        ((arrLen * nOut) + (IBLOCK_SIZE) - 1) & (~((IBLOCK_SIZE) - 1));
      const dim3 initblock = IBLOCK_SIZE;
      dim3 initgrid = initPaddedSize >> ( IBLOCK_SIZE_LOG2 );
      int nsteps = 1;
      while (initgrid.x > (1 << 14))
      {
          initgrid.x >>= 1;
          nsteps <<= 1;
          if (nsteps * initgrid.x * IBLOCK_SIZE < arrLen * nOut)
              initgrid.x++;
      }
      initKernel<<<initgrid,initblock,0,stream>>>(tmpOut, zero, arrLen * nOut, nsteps);
    }



    //int medExtShared = nOut;
    //const int shLimit = 0;
    //const int shLimit = 0;//16000 / 2;
    // Codepath below is a lot faster for random bins, a tad faster for real use-case
    // and a lot slower for degenerate key-distributions
#if USE_MEDIUM_PATH
    if (cuda_arch >= 120 && (histotype == histogram_atomic_inc || histotype == histogram_atomic_add))
    {
        const dim3 block = MEDIUM_BLOCK_SIZE;
        dim3 grid = nblocks;
        INDEXT nSteps = size / (INDEXT)( MEDIUM_BLOCK_SIZE * nblocks);

        INDEXT nFullSteps = 1;
        if (nSteps <= 0)
        {
            nFullSteps = 0;
            nblocks = (size >> MEDIUM_BLOCK_SIZE_LOG2);
            if ((nblocks << MEDIUM_BLOCK_SIZE_LOG2) < size) nblocks++;
        }
        if (nSteps > MAX_NLHSTEPS)
        {
            nFullSteps = size / ( MEDIUM_BLOCK_SIZE * nblocks * MAX_NLHSTEPS);
            nSteps = MAX_NLHSTEPS;
        }
        for (INDEXT step = 0; step < nFullSteps; step++)
        {
            histo_kernel_mediumNBins<histotype, nMultires><<<grid, block, 0, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, nSteps);
            start += (MEDIUM_BLOCK_SIZE * (INDEXT)nblocks * nSteps);
        }
        size = end - start;
        nSteps = size / (INDEXT)( MEDIUM_BLOCK_SIZE * nblocks);
        if (nSteps > 0)
        {
            histo_kernel_mediumNBins<histotype, nMultires><<<grid, block, 0, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, nSteps);
            start += (MEDIUM_BLOCK_SIZE * (INDEXT)nblocks * nSteps);
            size = end - start;
        }
        if (size > 0)
        {
            int ntblocks = size / ( MEDIUM_BLOCK_SIZE );
            if (ntblocks * MEDIUM_BLOCK_SIZE < size) ntblocks++;
            grid.x = ntblocks;
            histo_kernel_mediumNBins<histotype, nMultires><<<grid, block, 0, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, 1);
        }
    }
    else
#endif // USE_MEDIUM_PATH
    {
        INDEXT nFullSteps = 1;
        if (nSteps <= 0)
        {
            nFullSteps = 0;
            nblocks = (size >> LBLOCK_SIZE_LOG2);
            if ((nblocks << LBLOCK_SIZE_LOG2) < size) nblocks++;
        }
        if (nSteps > MAX_NLHSTEPS)
        {
            nFullSteps = size / ( LBLOCK_SIZE * (INDEXT)nblocks * MAX_NLHSTEPS);
            nSteps = MAX_NLHSTEPS;
        }
        for (int step = 0; step < nFullSteps; step++)
        {
            histo_kernel_largeNBins<histotype, nMultires><<<grid, block, extSharedNeeded, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, nSteps, hashSizelog2);
            start += (LBLOCK_SIZE * (INDEXT)nblocks * nSteps);
        }
        size = end - start;
        nSteps = size / ( LBLOCK_SIZE * (INDEXT)nblocks);
        if (nSteps > 0)
        {
            histo_kernel_largeNBins<histotype, nMultires><<<grid, block, extSharedNeeded, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, nSteps, hashSizelog2);
            start += (LBLOCK_SIZE * (INDEXT)nblocks * nSteps);
            size = end - start;
        }
        if (size > 0)
        {
            int ntblocks = size / ( LBLOCK_SIZE );
            if (ntblocks * LBLOCK_SIZE < size) ntblocks++;
            grid.x = ntblocks;
            histo_kernel_largeNBins<histotype, nMultires><<<grid, block, extSharedNeeded, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, 1, hashSizelog2);
        }
    }
#if H_ERROR_CHECKS
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
      printf("Cudaerror = %s\n", cudaGetErrorString( error ));
#endif

    // OK - so now tmpOut contains our gold - we just need to dig it out now

    enum cudaMemcpyKind fromOut = outInDev ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    enum cudaMemcpyKind toOut = outInDev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;

    if (stream != 0)
        cudaMemcpyAsync(&tmpOut[arrLen * nOut], out, sizeof(OUTPUTTYPE) * nOut, fromOut, stream);
    else
        cudaMemcpy(&tmpOut[arrLen * nOut], out, sizeof(OUTPUTTYPE) * nOut, fromOut);
    grid.x = nOut;
    //grid.x = nOut >> LBLOCK_SIZE_LOG2;
    //if ((grid.x << LBLOCK_SIZE_LOG2) < nOut) grid.x++;
    block.x = GATHER_BLOCK_SIZE;
    gatherKernel<<<grid, block, 0, stream>>>(sumfunObj, tmpOut, nOut, arrLen /** LBLOCK_WARPS*/, zero);
    // TODO: Async copy here also???
    if (outInDev && stream != 0)
        cudaMemcpyAsync(out, tmpOut, nOut*sizeof(OUTPUTTYPE), toOut, stream);
    else
        cudaMemcpy(out, tmpOut, nOut*sizeof(OUTPUTTYPE), toOut);

    // CPU-code path for debugging here:
/*    {
      int resIdx;
      int i;
      OUTPUTTYPE* h_tmp = (OUTPUTTYPE*)malloc(nblocks * nOut * sizeof(OUTPUTTYPE));
      //parallel_copy(h_tmp, MemType_HOST, tmpOut, MemType_DEV, n * nOut * sizeof(OUTPUTTYPE));
      cudaMemcpy(h_tmp, tmpOut, nblocks*nOut*sizeof(OUTPUTTYPE), cudaMemcpyDeviceToHost);
      for (resIdx = 0; resIdx < nOut; resIdx++)
      {
        OUTPUTTYPE res = out[resIdx];
        for (i = 0; i < nblocks; i++)
        {
          res = sumfunObj(res, h_tmp[i * nOut + resIdx]);
        }
        out[resIdx] = sumfunObj(res, out[resIdx]);
      }
      free(h_tmp);
    }
*/
    if (!tmpBuffer)
        cudaFree(tmpOut);

}




static int determineNKeySetsLog2(size_t size_out, int nOut, cudaDeviceProp* props)
{
    // 32 threads per block, one block shares one binset
    // Go for 2x occupancy = 64 active threads per block
    // Hence if we have NBinSets, then we need  tot_size x nOut x NBinSets x 2 bytes of shared
    // On sm_20 we have 48 000 bytes and on sm_1x 16 000
    // Hence nbinsets = SharedMem / (2 * tot_size * nOut)
    // For example sm_20, 16 int bins:
    //      nbinsets = 48000 / 2 * 4 * 16 = 48000 / 2*64 = 48000 / 128 = 375...
    // More than enough, but is it enough active threadblocks??
    int nBytesShared = 16000;
    size_t sizetot = size_out + sizeof(int);
    int nBinSets = nBytesShared / (sizetot * 2 * nOut);
// NOTE: Disabling for now - advantages seem nonexistent
//    if (nBinSets >= 32) return 5;
//    if (nBinSets >= 16) return 4;
//    if (nBinSets >= 8) return 3;
//    if (nBinSets >= 4) return 2;
//    if (nBinSets >= 2) return 1;
    if (nBinSets >= 1) return 0;
    return -1;
}





#if __CUDA_ARCH__ >= 200


template <int nMultires>
static inline __device__
bool checkForReduction (int* myKeys, int* rkeys)
{
    // Idea - if there is a large number of degenerate entries then we don't need to check them all for degeneracy
    // TODO: Implement the wonderful idea
    //return ((threadIdx.x >> 5) & 3) < 3;
#if 1
    bool myKeyDegenerate;
    //TAKE_WARP_MUTEX(0);
    rkeys[threadIdx.x & 31] = myKeys[0];
    // Check two thirds
    myKeyDegenerate =
        (myKeys[0] == (rkeys[(threadIdx.x + 1) & 31]))
	/*||
        (myKeys[0] == (rkeys[(threadIdx.x + 8) & 31]))*/;
    //GIVE_WARP_MUTEX(0);
    unsigned int degenMask = __ballot(myKeyDegenerate);
    // Estimate number of degenerate keys - if all are degenerate, the estimate is accurate
    int nDegen = __popc(degenMask);
    if (nDegen > HISTOGRAM_DEGEN_LIMIT)
        return true;
    else
        return false;
#endif
}
#endif



template <histogram_type histotype, int nBinSetslog2, int nMultires, bool laststeps, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
static inline __device__
void histogramKernel_stepImpl(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT end,
    OUTPUTTYPE zero,
    int nOut, INDEXT startidx,
    OUTPUTTYPE* bins, int* locks,
    OUTPUTTYPE* rvals, int* rkeys,
    int* doReduce, bool checkReduce,
    int* warpmutex)
{
    int myKeys[nMultires];
    OUTPUTTYPE vals[nMultires];
    bool doWrite = true;
    if (laststeps){
        if (startidx < end)
        {
            xformObj(input, startidx, &myKeys[0], &vals[0], nMultires);
        }
        else
        {
            doWrite = false;
            #pragma unroll
            for (int r = 0; r < nMultires; r++){
                vals[r] = zero;
                myKeys[r] = -1;
            }
        }
    }
    else
    {
        xformObj(input, startidx, &myKeys[0], &vals[0], nMultires);
    }
     // See keyIndex-reasoning above
    int binSet = (threadIdx.x & ((1 << nBinSetslog2) - 1));
#if __CUDA_ARCH__ >= 200
/*    if (laststeps){
        *doReduce = false;
    }
    else*/
    {
         if (checkReduce){
            *doReduce = checkForReduction<nMultires>(myKeys, rkeys);
            if (histotype == histogram_generic || histotype == histogram_atomic_add){
                __shared__ int tmp;
                tmp = 0;
                __syncthreads();
                if (*doReduce && ((threadIdx.x & 31) == 0)) atomicAdd(&tmp, 1);
                __syncthreads();
                if (tmp > HBLOCK_SIZE / 2)
                    *doReduce = true;
                else
                    *doReduce = false;
            }
            //if (laststeps) *doReduce = false;
/*            __syncthreads();
            bool tmpred = checkForReduction<nMultires>(myKeys, rkeys);
            if ((threadIdx.x & 31) == 0) atomicExch(doReduce, (int)tmpred);
            __syncthreads();*/
         }
    }
#endif

     // TODO: Unroll this later - nvcc (at least older versions) can't unroll atomics (?)
    // TODO: How to avoid bank-conflicts? Any way to avoid?


#if __CUDA_ARCH__ >= 200
#define ONE_HS_STEP(RESID) do { if ((RESID) < nMultires) { \
             int keyIndex = doWrite == false ? 0 : (myKeys[(RESID % nMultires)] << nBinSetslog2) + binSet; \
             if (*doReduce){\
                if (histotype == histogram_generic || histotype == histogram_atomic_add){\
                    bool Iwrite;\
                    TAKE_WARP_MUTEX(0);\
                    Iwrite = reduceToUnique<histotype, false>(&vals[(RESID % nMultires)], myKeys[(RESID % nMultires)], NULL, sumfunObj, rkeys, rvals);\
                    if (Iwrite && doWrite) bins[keyIndex] = sumfunObj(bins[keyIndex], vals[(RESID % nMultires)]);\
                    /*if (histotype == histogram_generic) myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, Iwrite && doWrite, warpmutex);\
                    else wrapAtomicAddWarp<laststeps>(&bins[keyIndex], *(&vals[(RESID % nMultires)]), &locks[keyIndex], sumfunObj, Iwrite && doWrite, warpmutex);*/\
                    GIVE_WARP_MUTEX(0);\
                } else { \
                    bool Iwrite = reduceToUnique<histotype, false>(&vals[(RESID % nMultires)], myKeys[(RESID % nMultires)], NULL, sumfunObj, rkeys, rvals);\
                    wrapAtomicAddWarp<laststeps>(&bins[keyIndex], *(&vals[(RESID % nMultires)]), &locks[keyIndex], sumfunObj, Iwrite && doWrite, warpmutex); \
                }\
             } else {\
                if (histotype == histogram_generic)\
                    myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, doWrite, warpmutex);\
                else if (histotype == histogram_atomic_add)\
                    wrapAtomicAddWarp<laststeps>(&bins[keyIndex], *(&vals[(RESID % nMultires)]), &locks[keyIndex], sumfunObj, doWrite, warpmutex);\
                else  if (histotype == histogram_atomic_inc)\
                    wrapAtomicIncWarp<laststeps>(&bins[keyIndex], &locks[keyIndex], sumfunObj, doWrite, warpmutex);\
                else{\
                    myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, doWrite, warpmutex);\
                 }\
             } } } while (0)
#else
#define ONE_HS_STEP(RESID) do { if ((RESID) < nMultires) { \
                int keyIndex = doWrite == false ? 0 : (myKeys[(RESID % nMultires)] << nBinSetslog2) + binSet; \
                if (histotype == histogram_generic)\
                    myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, doWrite, warpmutex);\
                else if (histotype == histogram_atomic_add)\
                    wrapAtomicAddWarp<laststeps>(&bins[keyIndex], *(&vals[(RESID % nMultires)]), &locks[keyIndex], sumfunObj, doWrite, warpmutex);\
                else  if (histotype == histogram_atomic_inc)\
                    wrapAtomicIncWarp<laststeps>(&bins[keyIndex], &locks[keyIndex], sumfunObj, doWrite, warpmutex);\
                else{\
                    myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, doWrite, warpmutex);\
                 }\
                } } while (0)
#endif

     ONE_HS_STEP(0);
     ONE_HS_STEP(1);
     ONE_HS_STEP(2);
     ONE_HS_STEP(3);
     //#pragma unroll
     for (int resid = 4; resid < nMultires; resid++){
         ONE_HS_STEP(resid);
     }
#undef ONE_HS_STEP
}


template <int nBinSetslog2, histogram_type histotype, int nMultires, bool lastSteps, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
__global__
void histogramKernel_sharedbins_new(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero,
    OUTPUTTYPE* blockOut, int nOut,
    int outStride,
    int nSteps)
{
  extern __shared__ int cudahistogram_binstmp[];
  OUTPUTTYPE* bins = (OUTPUTTYPE*)&(*cudahistogram_binstmp);

  int* locks = (int*)&bins[(nOut << nBinSetslog2)];
  int* rkeys = NULL;
  OUTPUTTYPE* rvals = NULL;
  //__shared__
  int warpmutex;
  //INIT_WARP_MUTEX2(warpmutex);

#if __CUDA_ARCH__ >= 200
  int warpId = threadIdx.x >> 5;
  if (histotype == histogram_generic)
      rkeys = &locks[(nOut << nBinSetslog2)];
  else
      rkeys = locks;
  rvals = (OUTPUTTYPE*)&rkeys[32];
  if (histotype == histogram_atomic_inc){
      rkeys = &rkeys[warpId << 5];
      //rvals = &rvals[warpId << 5];
  }
#endif
  const int nBinSets = 1 << nBinSetslog2;
  // Reset all bins to zero...
  for (int j = 0; j < ((nOut << nBinSetslog2) >> HBLOCK_SIZE_LOG2) + 1; j++)
  {
    int bin = (j << HBLOCK_SIZE_LOG2) + threadIdx.x;
    if (bin < (nOut << nBinSetslog2)){
        bins[bin] = zero;
    }
  }
#if HBLOCK_SIZE > 32
  __syncthreads();
#endif
  int outidx = blockIdx.x;
  INDEXT startidx = (INDEXT)((outidx * nSteps) * HBLOCK_SIZE + start + threadIdx.x);

  /*__shared__*/ int doReduce; // local var - TODO: Is this safe??
  doReduce = 0;

#define MED_UNROLL_LOG2     2
#define MED_UNROLL          (1 << MED_UNROLL_LOG2)
  int step;
  for (step = 0; step < (nSteps >> MED_UNROLL_LOG2); step++)
  {
      //#pragma unroll
      //for (int substep = 0; substep < MED_UNROLL; substep++){
          histogramKernel_stepImpl<histotype, nBinSetslog2, nMultires, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
            (input, xformObj, sumfunObj, end, zero, nOut, startidx, bins, locks, rvals, rkeys, &doReduce, true, &warpmutex);
          startidx += HBLOCK_SIZE;
          histogramKernel_stepImpl<histotype, nBinSetslog2, nMultires, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
            (input, xformObj, sumfunObj, end, zero, nOut, startidx, bins, locks, rvals, rkeys, &doReduce, false, &warpmutex);
          startidx += HBLOCK_SIZE;
          histogramKernel_stepImpl<histotype, nBinSetslog2, nMultires, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
            (input, xformObj, sumfunObj, end, zero, nOut, startidx, bins, locks, rvals, rkeys, &doReduce, false, &warpmutex);
          startidx += HBLOCK_SIZE;
          histogramKernel_stepImpl<histotype, nBinSetslog2, nMultires, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
            (input, xformObj, sumfunObj, end, zero, nOut, startidx, bins, locks, rvals, rkeys, &doReduce, false, &warpmutex);
          startidx += HBLOCK_SIZE;
      //}
  }
  step = (nSteps >> MED_UNROLL_LOG2) << MED_UNROLL_LOG2;
  for (; step < nSteps ; step++)
  {
      histogramKernel_stepImpl<histotype, nBinSetslog2, nMultires, lastSteps, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
        (input, xformObj, sumfunObj, end, zero, nOut, startidx, bins, locks, rvals, rkeys, &doReduce, (step & 7) == 0, &warpmutex);
      startidx += HBLOCK_SIZE;
  }
#undef MED_UNROLL
#undef MED_UNROLL_LOG2
#if HBLOCK_SIZE > 32
  __syncthreads();
#endif
  // Finally put together the bins
  for (int j = 0; j < (nOut >> HBLOCK_SIZE_LOG2) + 1; j++) {
    int key = (j << HBLOCK_SIZE_LOG2) + threadIdx.x;
    if (key < nOut)
    {
      OUTPUTTYPE res = blockOut[key * outStride + outidx];
      //int tmpBin = bin;
#pragma unroll
      for (int k = 0; k < nBinSets; k++)
      {
          //tmpBin += nOut;
          res = sumfunObj(res, bins[(key << nBinSetslog2) + k]);
      }
        //printf("tid:%02d, write out bin: %02d, \n", threadIdx.x, bin);
      blockOut[key * outStride + outidx] = res;
    }
  }
}


template <histogram_type histotype, typename OUTPUTTYPE>
static int getMediumHistoTmpbufSize(int nOut, cudaDeviceProp* props)
{
    int nblocks = props->multiProcessorCount * 8;
    // NOTE: The other half is used by multireduce...
    return 2 * nblocks * nOut * sizeof(OUTPUTTYPE);
}



template <histogram_type histotype, int nMultires,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
static
void callHistogramKernelImpl(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero, OUTPUTTYPE* out, int nOut,
    cudaDeviceProp* props,
    cudaStream_t stream,
    size_t* getTmpBufSize,
    void* tmpBuffer,
    bool outInDev,
    int cuda_arch)
{
  INDEXT size = end - start;
  // Check if there is something to do actually...
  if (end <= start)
  {
      if (getTmpBufSize) *getTmpBufSize = 0;
      return;
  }
  int nblocks = props->multiProcessorCount * 8;


  // Assert that our grid is not too large!
  //MY_ASSERT(n < 65536 && "Sorry - currently we can't do such a big problems with histogram-kernel...");
  // One entry for each output for each thread-block:
  //OUTPUTTYPE* tmpOut = (OUTPUTTYPE*)parallel_alloc(MemType_DEV, n * nOut * sizeof(OUTPUTTYPE));
  OUTPUTTYPE* tmpOut;
  if (getTmpBufSize)
  {
      // NOTE: The other half is used by multireduce...
      *getTmpBufSize = 2 * nblocks * nOut * sizeof(OUTPUTTYPE);
      return;
  }

  int nsteps = size / ( nblocks * HBLOCK_SIZE );
  if (nsteps *  nblocks * HBLOCK_SIZE  < size) nsteps++;
  if (nsteps > MAX_NHSTEPS)
      nsteps = MAX_NHSTEPS;


  if (tmpBuffer)
  {
    char* tmpptr = (char*)tmpBuffer;
    tmpOut = (OUTPUTTYPE*)tmpBuffer;
    tmpBuffer = (void*)&tmpptr[nblocks * nOut * sizeof(OUTPUTTYPE)];
  }
  else
  {
    cudaMalloc((void**)&tmpOut, nblocks * nOut * sizeof(OUTPUTTYPE));
  }

  /* For block size other that power of two:
   const dim3 grid = size / BLOCK_SIZE +
                   ( size % BLOCK_SIZE == 0 ? 0 : 1 );
   */
  //MY_ASSERT(size > 0);
  //cudaMemsetAsync(tmpOut, 0xFF, n * nOut * sizeof(OUTPUTTYPE), CURRENT_STREAM() );
  //cudaMemset(tmpOut, 0xFF, n * nOut * sizeof(OUTPUTTYPE) );
  {
#define IBLOCK_SIZE_LOG2    7
#define IBLOCK_SIZE         (1 << IBLOCK_SIZE_LOG2)
    int initPaddedSize =
      ((nblocks * nOut) + (IBLOCK_SIZE) - 1) & (~((IBLOCK_SIZE) - 1));
    const dim3 initblock = IBLOCK_SIZE;
    dim3 initgrid = initPaddedSize >> ( IBLOCK_SIZE_LOG2 );
    int nsteps = 1;
    while (initgrid.x > (1 << 14))
    {
        initgrid.x >>= 1;
        nsteps <<= 1;
        if (nsteps * initgrid.x * IBLOCK_SIZE < nblocks * nOut)
            initgrid.x++;
    }
    initKernel<<<initgrid,initblock,0,stream>>>(tmpOut, zero, nblocks * nOut, nsteps);
#undef IBLOCK_SIZE_LOG2
#undef IBLOCK_SIZE
  }
  int nKeysetslog2 = determineNKeySetsLog2(sizeof(OUTPUTTYPE), nOut, props);
  if (nKeysetslog2 < 0) nKeysetslog2 = 0;
  int extSharedNeeded = ((nOut << nKeysetslog2)) * (sizeof(OUTPUTTYPE)); // bins
  if (histotype == histogram_generic || cuda_arch < 130)
      extSharedNeeded += ((nOut << nKeysetslog2)) * (sizeof(int)); // locks

  if (cuda_arch >= 200)
  {
      // Reduction stuff:
      if (histotype == histogram_generic || histotype == histogram_atomic_add)
      {
          extSharedNeeded += ((sizeof(OUTPUTTYPE) + sizeof(int)) << 5); // reduction values
      }
      else
      {
          extSharedNeeded += (sizeof(int) << HBLOCK_SIZE_LOG2); // keys per warp of one thread
      }

  }
  /*int extSharedNeeded = ((nOut << nKeysetslog2)) * (sizeof(OUTPUTTYPE) + sizeof(int)) + (sizeof(OUTPUTTYPE) * HBLOCK_SIZE);
  if (nOut < HBLOCK_SIZE) extSharedNeeded += sizeof(int) * (HBLOCK_SIZE - nOut);
  if (cuda_arch < 130)
      extSharedNeeded += ((nOut << nKeysetslog2)) * (sizeof(int));*/
  //printf("binsets = %d, steps = %d\n", (1 << nKeysetslog2), nsteps);
  int nOrigBlocks = nblocks;
  INDEXT myStart = start;
  while(myStart < end)
  {
      bool lastStep = false;
      if (myStart + nsteps * nblocks * HBLOCK_SIZE > end)
      {
          size = end - myStart;
          nsteps = (size) / (nblocks * HBLOCK_SIZE);
          if (nsteps < 1)
          {
              lastStep = true;
              nsteps = 1;
              nblocks = size / HBLOCK_SIZE;
              if (nblocks * HBLOCK_SIZE < size)
                  nblocks++;
          }
      }
      dim3 grid = nblocks;
      dim3 block = HBLOCK_SIZE;
      switch (nKeysetslog2)
      {
        case 0:
          if (lastStep)
            histogramKernel_sharedbins_new<0, histotype, nMultires, true><<<grid, block, extSharedNeeded, stream>>>(
              input, xformObj, sumfunObj, myStart, end, zero, tmpOut, nOut, nOrigBlocks, nsteps);
          else
              histogramKernel_sharedbins_new<0, histotype, nMultires, false><<<grid, block, extSharedNeeded, stream>>>(
                input, xformObj, sumfunObj, myStart, end, zero, tmpOut, nOut, nOrigBlocks, nsteps);
          break;
    /*    case 1:
          histogramKernel_sharedbins_new<1, histotype, nMultires><<<grid, block, extSharedNeeded, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, n, nsteps);
          break;
        case 2:
          histogramKernel_sharedbins_new<2, histotype, nMultires><<<grid, block, extSharedNeeded, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, n, nsteps);
          break;
        case 3:
          histogramKernel_sharedbins_new<3, histotype, nMultires><<<grid, block, extSharedNeeded, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, n, nsteps);
          break;
        case 4:
          histogramKernel_sharedbins_new<4, histotype, nMultires><<<grid, block, extSharedNeeded, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, n, nsteps);
          break;
        case 5:
          histogramKernel_sharedbins_new<5, histotype, nMultires><<<grid, block, extSharedNeeded, stream>>>(
              input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, n, nsteps);
          break;*/
        case -1:
            // TODO: Error?
          //assert(0); // "Sorry - not implemented yet"
            break;
      }
      myStart += nsteps * nblocks * HBLOCK_SIZE;
  }

#if H_ERROR_CHECKS
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("Cudaerror = %s\n", cudaGetErrorString( error ));
#endif
  // OK - so now tmpOut contains our gold - we just need to dig it out now

  callMultiReduce(nOrigBlocks, nOut, out, tmpOut, sumfunObj, zero, stream, tmpBuffer, outInDev);
  // Below same as host-code
  #if 0
  {
    int resIdx;
    int i;
    OUTPUTTYPE* h_tmp = (OUTPUTTYPE*)malloc(n * nOut * sizeof(OUTPUTTYPE));
    //parallel_copy(h_tmp, MemType_HOST, tmpOut, MemType_DEV, n * nOut * sizeof(OUTPUTTYPE));
    cudaMemcpy(h_tmp, tmpOut, n*nOut*sizeof(OUTPUTTYPE), cudaMemcpyDeviceToHost);
    for (resIdx = 0; resIdx < nOut; resIdx++)
    {
      OUTPUTTYPE res = out[resIdx];
      for (i = 0; i < n; i++)
      {
        res = sumfunObj(res, h_tmp[i + resIdx * n]);
      }
      out[resIdx] = res;
    }
    free(h_tmp);
  }
  #endif
  //parallel_free(tmpOut, MemType_DEV);
  if (!tmpBuffer)
    cudaFree(tmpOut);
}

template <typename OUTTYPE>
static
bool binsFitIntoShared(int nOut, OUTTYPE zero, cudaDeviceProp* props, int cuda_arch)
{
  // Assume here we can only use 16kb of shared in total per SM
  // Also lets take minimal of 2 threads per functional unit active, in
  // order to be able to hide at least some latencies - for Fermi this means 32 * 2 = 64
  // of active threads needed in total (Note: This is minimal and will hurt perf).
  // Also we run blocks of 32 threads and each block needs its own bin - therefore
  // we need in total 2 full bin-sets per SM plus 32 bins for the one for the working part
  // of the algorithm.

  // Due to these considerations we infer that we can fit it nicely in, if
  // (4 binsets x Nbins/binset + 32) x sizeof(OUTYPE) < 16kib - let's take here 16kb to have some room
  // for required parameters

  // Example: 64 doubles: 8bytes per number double => (4 * 64 + 32) * 8bytes = 288 * 8 bytes = 2304 bytes -> Easy

  // How many bins of doubles can we do with these limits?
  //    ( 4 * x + 32) * 8bytes = 16000 bytes <=> 4x = 2000 - 32 => x = 2000/4 - 32/4 = 500 - 8 = 492 bins.

  // TODO: A possibly faster version of this would be to share one set of bins over as many warps as possible
  // for example, if we would use 512 threads = 16 warps, then this would be fine for hiding probably all major latencies
  // and we could get away with just one binset on SM:

  // ( x + 512 ) * 8bytes = 16000 bytes <=> x = 2000 - 512 = 1488 bins! With better latency-hiding
  // On the other hand this requires atomic operations on the shared memory, which could be somewhat slower on
  // arbitrary types, but all in all, this would seem to provide a better route. At least worth investigating...
  int shlimit = props->sharedMemPerBlock - 300;
  int limit = shlimit;
  // TODO: Pessimistic limit
  int need = (sizeof(zero) + sizeof(int)) * nOut;
  if (cuda_arch >= 200)
    need += HBLOCK_SIZE * sizeof(int) + 32 * sizeof(zero);
  if (need <= limit)
    return true;
  return false;
}

template <bool subHisto, histogram_type histotype, int nBinSetslog2, int nMultires, bool laststeps, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
static inline __device__
void histogramKernel_stepImplMulti(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT end,
    OUTPUTTYPE zero,
    int subsize, INDEXT startidx,
    OUTPUTTYPE* bins, int* locks,
    OUTPUTTYPE* rvals, int* rkeys,
    int* doReduce, bool checkReduce,
    int* warpmutex, int binOffset)
{
    int myKeys[nMultires];
    OUTPUTTYPE vals[nMultires];
    bool doWrite = true;
    if (laststeps){
        if (startidx < end)
        {
            xformObj(input, startidx, &myKeys[0], &vals[0], nMultires);
        }
        else
        {
            doWrite = false;
            #pragma unroll
            for (int r = 0; r < nMultires; r++){
                vals[r] = zero;
                myKeys[r] = -1;
            }
        }
    }
    else
    {
        xformObj(input, startidx, &myKeys[0], &vals[0], nMultires);
    }
#if __CUDA_ARCH__ >= 200
/*    if (laststeps){
        *doReduce = false;
    }
    else*/
    {
         if (checkReduce){
            *doReduce = checkForReduction<nMultires>(myKeys, rkeys);
            if (histotype == histogram_generic || histotype == histogram_atomic_add){
                __shared__ int tmp;
                tmp = 0;
                __syncthreads();
                if (*doReduce && ((threadIdx.x & 31) == 0)) atomicAdd(&tmp, 1);
                __syncthreads();
                if (tmp > HMBLOCK_SIZE / 2)
                    *doReduce = true;
                else
                    *doReduce = false;
            }
            //if (laststeps) *doReduce = false;
/*            __syncthreads();
            bool tmpred = checkForReduction<nMultires>(myKeys, rkeys);
            if ((threadIdx.x & 31) == 0) atomicExch(doReduce, (int)tmpred);
            __syncthreads();*/
         }
    }
#endif

     // TODO: Unroll this later - nvcc (at least older versions) can't unroll atomics (?)
    // TODO: How to avoid bank-conflicts? Any way to avoid?


#if __CUDA_ARCH__ >= 200
#define ONE_HS_STEP(RESID) do { if ((RESID) < nMultires) { \
             int keyIndex = (myKeys[(RESID % nMultires)] - binOffset); \
             bool Iwrite = keyIndex >= 0 && keyIndex < subsize && doWrite;\
             if (!Iwrite) keyIndex = 0; \
             if (*doReduce){\
                if (histotype == histogram_generic || histotype == histogram_atomic_add){\
                    TAKE_WARP_MUTEX(0);\
                    bool Iwrite2 = reduceToUnique<histotype, false>(&vals[(RESID % nMultires)], myKeys[(RESID % nMultires)], NULL, sumfunObj, rkeys, rvals);\
                    if (Iwrite && Iwrite2) \
                        bins[keyIndex] = sumfunObj(bins[keyIndex], vals[(RESID % nMultires)]);\
                    /*if (histotype == histogram_generic) myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, Iwrite && doWrite, warpmutex);\
                    else wrapAtomicAddWarp<laststeps>(&bins[keyIndex], *(&vals[(RESID % nMultires)]), &locks[keyIndex], sumfunObj, Iwrite && doWrite, warpmutex);*/\
                    GIVE_WARP_MUTEX(0);\
                } else { \
                    bool Iwrite2 = reduceToUnique<histotype, false>(&vals[(RESID % nMultires)], myKeys[(RESID % nMultires)], NULL, sumfunObj, rkeys, rvals);\
                    wrapAtomicAddWarp<laststeps>(&bins[keyIndex], *(&vals[(RESID % nMultires)]), &locks[keyIndex], sumfunObj, Iwrite && Iwrite2, warpmutex); \
                }\
             } else {\
                if (!Iwrite) keyIndex = 0;\
                if (histotype == histogram_generic)\
                    myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, Iwrite, warpmutex);\
                else if (histotype == histogram_atomic_add)\
                    wrapAtomicAddWarp<laststeps>(&bins[keyIndex], *(&vals[(RESID % nMultires)]), &locks[keyIndex], sumfunObj, Iwrite, warpmutex);\
                else  if (histotype == histogram_atomic_inc)\
                    wrapAtomicIncWarp<laststeps>(&bins[keyIndex], &locks[keyIndex], sumfunObj, Iwrite, warpmutex);\
                else{\
                    myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, Iwrite, warpmutex);\
                 }\
             } } } while (0)
#else
#define ONE_HS_STEP(RESID) do { if ((RESID) < nMultires) { \
                int keyIndex = (myKeys[(RESID % nMultires)] - binOffset); \
                bool Iwrite = keyIndex >= 0 && keyIndex < subsize && doWrite;\
                if (!Iwrite) keyIndex = 0;\
                if (histotype == histogram_generic)\
                    myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, Iwrite, warpmutex);\
                else if (histotype == histogram_atomic_add)\
                    wrapAtomicAddWarp<laststeps>(&bins[keyIndex], *(&vals[(RESID % nMultires)]), &locks[keyIndex], sumfunObj, Iwrite, warpmutex);\
                else  if (histotype == histogram_atomic_inc)\
                    wrapAtomicIncWarp<laststeps>(&bins[keyIndex], &locks[keyIndex], sumfunObj, Iwrite, warpmutex);\
                else{\
                    myAtomicWarpAdd<laststeps>(&bins[keyIndex], vals[(RESID % nMultires)], &locks[keyIndex], sumfunObj, Iwrite, warpmutex);\
                 }\
                } } while (0)
#endif

     ONE_HS_STEP(0);
     ONE_HS_STEP(1);
     ONE_HS_STEP(2);
     ONE_HS_STEP(3);
     //#pragma unroll
     for (int resid = 4; resid < nMultires; resid++){
         ONE_HS_STEP(resid);
     }
#undef ONE_HS_STEP


}


template <histogram_type histotype, int nMultires, bool lastSteps, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
__global__
void histogramKernel_multipass(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero,
    OUTPUTTYPE* blockOut, int nOut,
    int outStride,
    int nSteps,
    int subsize)
{
  extern __shared__ int cudahistogram_binstmp[];
  OUTPUTTYPE* bins = (OUTPUTTYPE*)&(*cudahistogram_binstmp);

  int* locks = (int*)&bins[subsize];
  int* rkeys = NULL;
  OUTPUTTYPE* rvals = NULL;
  //__shared__
  int warpmutex;
  //INIT_WARP_MUTEX2(warpmutex);

#if __CUDA_ARCH__ >= 200
  int warpId = threadIdx.x >> 5;
  if (histotype == histogram_generic)
      rkeys = &locks[subsize];
  else
      rkeys = locks;
  rvals = (OUTPUTTYPE*)&rkeys[32];
  if (histotype == histogram_atomic_inc){
      rkeys = &rkeys[warpId << 5];
      //rvals = &rvals[warpId << 5];
  }
#endif
  // Reset all bins to zero...
  for (int j = 0; j < (subsize >> HMBLOCK_SIZE_LOG2) + 1; j++)
  {
    int bin = (j << HMBLOCK_SIZE_LOG2) + threadIdx.x;
    if (bin < subsize){
        bins[bin] = zero;
    }
  }
#if HMBLOCK_SIZE > 32
  __syncthreads();
#endif
  int outidx = blockIdx.y;
  int binOffset = blockIdx.x * subsize;
  INDEXT startidx = (INDEXT)((outidx * nSteps) * HMBLOCK_SIZE + start + threadIdx.x);

  int doReduce; // local var - TODO: Is this safe??
  doReduce = 0;

#define MED_UNROLL_LOG2     2
#define MED_UNROLL          (1 << MED_UNROLL_LOG2)
  int step;
  for (step = 0; step < (nSteps >> MED_UNROLL_LOG2); step++)
  {
      histogramKernel_stepImplMulti<true, histotype, 0, nMultires, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
        (input, xformObj, sumfunObj, end, zero, subsize, startidx, bins, locks, rvals, rkeys, &doReduce, true, &warpmutex, binOffset);
      startidx += HMBLOCK_SIZE;
      histogramKernel_stepImplMulti<true, histotype, 0, nMultires, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
        (input, xformObj, sumfunObj, end, zero, subsize, startidx, bins, locks, rvals, rkeys, &doReduce, false, &warpmutex, binOffset);
      startidx += HMBLOCK_SIZE;
      histogramKernel_stepImplMulti<true, histotype, 0, nMultires, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
        (input, xformObj, sumfunObj, end, zero, subsize, startidx, bins, locks, rvals, rkeys, &doReduce, false, &warpmutex, binOffset);
      startidx += HMBLOCK_SIZE;
      histogramKernel_stepImplMulti<true, histotype, 0, nMultires, false, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
        (input, xformObj, sumfunObj, end, zero, subsize, startidx, bins, locks, rvals, rkeys, &doReduce, false, &warpmutex, binOffset);
      startidx += HMBLOCK_SIZE;
  }
  step = (nSteps >> MED_UNROLL_LOG2) << MED_UNROLL_LOG2;
  for (; step < nSteps ; step++)
  {
      histogramKernel_stepImplMulti<true, histotype, 0, nMultires, lastSteps, INPUTTYPE, TRANSFORMFUNTYPE, SUMFUNTYPE, OUTPUTTYPE>
        (input, xformObj, sumfunObj, end, zero, subsize, startidx, bins, locks, rvals, rkeys, &doReduce, (step & 7) == 0, &warpmutex, binOffset);
      startidx += HMBLOCK_SIZE;
  }
#undef MED_UNROLL
#undef MED_UNROLL_LOG2

#if HMBLOCK_SIZE > 32
  __syncthreads();
#endif
  // Finally put together the bins
  for (int j = 0; j < (subsize >> HMBLOCK_SIZE_LOG2) + 1; j++) {
    int key = (j << HMBLOCK_SIZE_LOG2) + threadIdx.x;
    if (key < subsize)
    {
      OUTPUTTYPE res = blockOut[(key + binOffset) * outStride + outidx];
      //int tmpBin = bin;
      res = sumfunObj(res, bins[key]);
        //printf("tid:%02d, write out bin: %02d, \n", threadIdx.x, bin);
      blockOut[(key + binOffset) * outStride + outidx] = res;
    }
  }
}


static int determineSubHistoSize(int nOut, size_t outsize, histogram_type histotype, int cuda_arch, cudaDeviceProp* props)
{
    int shlimit = props->sharedMemPerBlock - 300;
    int neededPerKey = outsize;
    if (histotype == histogram_generic || cuda_arch < 130)
        neededPerKey += (sizeof(int)); // locks

    int neededConst = 0;

    if (cuda_arch >= 200)
    {
        // Reduction stuff:
        if (histotype == histogram_generic || histotype == histogram_atomic_add)
        {
            neededConst += (outsize + sizeof(int)) << 5; // reduction values
        }
        else
        {
            neededConst += (sizeof(int) << HMBLOCK_SIZE_LOG2); // keys per warp of one thread
        }
    }
    int result = (shlimit - neededConst) / (2*neededPerKey);
    int res = 0;
    if (result >= 1<<16) { result >>= 16; res += 16; }
    if (result >= 1<< 8) { result >>=  8; res +=  8; }
    if (result >= 1<< 4) { result >>=  4; res +=  4; }
    if (result >= 1<< 2) { result >>=  2; res +=  2; }
    if (result >= 1<< 1) {                res +=  1; }
    return (1 << res);
}

template <histogram_type histotype, typename OUTPUTTYPE>
static int getMultipassBufSize(int nOut, cudaDeviceProp* props, int cuda_arch)
{
    int subsize = determineSubHistoSize(nOut, sizeof(OUTPUTTYPE), histotype, cuda_arch, props);
    int nDegenBlocks = nOut / subsize;
    if (subsize * nDegenBlocks < nOut) nDegenBlocks++;
    int nblocks = props->multiProcessorCount;
    if (nDegenBlocks < 8)
        nblocks = props->multiProcessorCount * 8 / nDegenBlocks;

    //int nblocks = props->multiProcessorCount * 8;
    // NOTE: The other half is used by multireduce...
    //printf("getMultipassBufSize(%d) = %d\n", nOut, 2 * nblocks * nOut * sizeof(OUTPUTTYPE));
    return 2 * nblocks * nOut * sizeof(OUTPUTTYPE);
}



template <histogram_type histotype, int nMultires,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
static
void callHistogramKernelMultiPass(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero, OUTPUTTYPE* out, int nOut,
    cudaDeviceProp* props,
    cudaStream_t stream,
    void* tmpBuffer,
    bool outInDev,
    int cuda_arch)
{
  INDEXT size = end - start;
  if (end <= start)
      return;

  //int debugs = 0;

  int subsize = determineSubHistoSize(nOut, sizeof(OUTPUTTYPE), histotype, cuda_arch, props);
  int nDegenBlocks = nOut / subsize;
  if (subsize * nDegenBlocks < nOut) nDegenBlocks++;
  int nblocks = props->multiProcessorCount;
  if (nDegenBlocks < 8)
      nblocks = props->multiProcessorCount * 8 / nDegenBlocks;
  OUTPUTTYPE* tmpOut;

  int nsteps = size / ( nblocks * HMBLOCK_SIZE );
  if (nsteps *  nblocks * HMBLOCK_SIZE  < size) nsteps++;
  if (nsteps > MAX_MULTISTEPS)
      nsteps = MAX_MULTISTEPS;

  //printf(" <debugstep = %d> ", debugs++);

  bool userBuffer = false;

  if (tmpBuffer)
  {
    char* tmpptr = (char*)tmpBuffer;
    tmpOut = (OUTPUTTYPE*)tmpBuffer;
    tmpBuffer = (void*)&tmpptr[nblocks * nOut * sizeof(OUTPUTTYPE)];
    userBuffer = true;
    //printf("tmpBuffer =  &tmpptr[%d]\n", nblocks * nOut * sizeof(OUTPUTTYPE));
  }
  else
  {
    cudaMalloc((void**)&tmpOut, 2 * nblocks * nOut * sizeof(OUTPUTTYPE));
    //printf("tmpOut =  malloc(%d)\n", 2 * nblocks * nOut * sizeof(OUTPUTTYPE));
    //tmpBuffer = (void*)&tmpOut[nblocks * nOut * sizeof(OUTPUTTYPE)];
    //printf("tmpBuffer =  &tmpOut[%d]\n", nblocks * nOut * sizeof(OUTPUTTYPE));
  }

#define IBLOCK_SIZE_LOG2    7
#define IBLOCK_SIZE         (1 << IBLOCK_SIZE_LOG2)
    int initPaddedSize =
      ((nblocks * nOut) + (IBLOCK_SIZE) - 1) & (~((IBLOCK_SIZE) - 1));
    const dim3 initblock = IBLOCK_SIZE;
    dim3 initgrid = initPaddedSize >> ( IBLOCK_SIZE_LOG2 );
    int nsteps2 = 1;
    while (initgrid.x > (1 << 14))
    {
        initgrid.x >>= 1;
        nsteps2 <<= 1;
        if (nsteps2 * initgrid.x * IBLOCK_SIZE < nblocks * nOut)
            initgrid.x++;
    }
    initKernel<<<initgrid,initblock,0,stream>>>(tmpOut, zero, nblocks * nOut, nsteps2);
#undef IBLOCK_SIZE_LOG2
#undef IBLOCK_SIZE
  int extSharedNeeded = subsize * (sizeof(OUTPUTTYPE)); // bins
  if (histotype == histogram_generic || cuda_arch < 130)
      extSharedNeeded += subsize * (sizeof(int)); // locks

  if (cuda_arch >= 200)
  {
      // Reduction stuff:
      if (histotype == histogram_generic || histotype == histogram_atomic_add)
      {
          extSharedNeeded += ((sizeof(OUTPUTTYPE) + sizeof(int)) << 5); // reduction values
      }
      else
      {
          extSharedNeeded += (sizeof(int) << HMBLOCK_SIZE_LOG2); // keys per warp of one thread
      }

  }

  //printf(" <debugstep(init) = %d> ", debugs++);

  /*int extSharedNeeded = ((nOut << nKeysetslog2)) * (sizeof(OUTPUTTYPE) + sizeof(int)) + (sizeof(OUTPUTTYPE) * HMBLOCK_SIZE);
  if (nOut < HMBLOCK_SIZE) extSharedNeeded += sizeof(int) * (HMBLOCK_SIZE - nOut);
  if (cuda_arch < 130)
      extSharedNeeded += ((nOut << nKeysetslog2)) * (sizeof(int));*/
  //printf("binsets = %d, steps = %d\n", (1 << nKeysetslog2), nsteps);
  int nOrigBlocks = nblocks;
  INDEXT myStart = start;
  while(myStart < end)
  {
      bool lastStep = false;
      if (myStart + nsteps * nblocks * HMBLOCK_SIZE > end)
      {
          size = end - myStart;
          nsteps = (size) / (nblocks * HMBLOCK_SIZE);
          if (nsteps < 1)
          {
              lastStep = true;
              nsteps = 1;
              nblocks = size / HMBLOCK_SIZE;
              if (nblocks * HMBLOCK_SIZE < size)
                  nblocks++;
          }
      }
      dim3 grid;
      grid.y = nblocks;
      grid.x = nDegenBlocks;
      dim3 block = HMBLOCK_SIZE;
      //printf(" <debugstep(main) = %d> ", debugs++);
      if (lastStep)
        histogramKernel_multipass<histotype, nMultires, true><<<grid, block, extSharedNeeded, stream>>>(
          input, xformObj, sumfunObj, myStart, end, zero, tmpOut, nOut, nOrigBlocks, nsteps, subsize);
      else
          histogramKernel_multipass<histotype, nMultires, false><<<grid, block, extSharedNeeded, stream>>>(
            input, xformObj, sumfunObj, myStart, end, zero, tmpOut, nOut, nOrigBlocks, nsteps, subsize);
      myStart += nsteps * nblocks * HMBLOCK_SIZE;
  }

#if H_ERROR_CHECKS
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("Cudaerror = %s\n", cudaGetErrorString( error ));
#endif
  // OK - so now tmpOut contains our gold - we just need to dig it out now
  //printf(" <debugstep(out) = %d> ", debugs++);
  //printf("callMultiReduce(%d, %d,...)\n", nOrigBlocks, nOut);
  callMultiReduce(nOrigBlocks, nOut, out, tmpOut, sumfunObj, zero, stream, tmpBuffer, outInDev);
  //printf(" <debugstep(multireduce) = %d> ", debugs++);
#if H_ERROR_CHECKS
    error = cudaGetLastError();
    if (error != cudaSuccess)
        printf("Cudaerror(reduce) = %s\n", cudaGetErrorString( error ));
#endif

  // Below same as host-code
  #if 0
  {
    int resIdx;
    int i;
    OUTPUTTYPE* h_tmp = (OUTPUTTYPE*)malloc(n * nOut * sizeof(OUTPUTTYPE));
    //parallel_copy(h_tmp, MemType_HOST, tmpOut, MemType_DEV, n * nOut * sizeof(OUTPUTTYPE));
    cudaMemcpy(h_tmp, tmpOut, n*nOut*sizeof(OUTPUTTYPE), cudaMemcpyDeviceToHost);
    for (resIdx = 0; resIdx < nOut; resIdx++)
    {
      OUTPUTTYPE res = out[resIdx];
      for (i = 0; i < n; i++)
      {
        res = sumfunObj(res, h_tmp[i + resIdx * n]);
      }
      out[resIdx] = res;
    }
    free(h_tmp);
  }
  #endif
  //parallel_free(tmpOut, MemType_DEV);
  if (!userBuffer)
    cudaFree(tmpOut);
}







template <bool lastSteps, int nMultires,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE, typename OUTPUTTYPE, typename INDEXT>
static inline __device__
void histoKernel_smallBinStep(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT myStart, INDEXT end,
    OUTPUTTYPE* mySHBins)
{
    int myKeys[nMultires];
    if (lastSteps)
    {
        if (myStart < end)
        {
            OUTPUTTYPE myOut[nMultires];
            xformObj(input, myStart, &myKeys[0], &myOut[0], nMultires);
#pragma unroll
            for (int res = 0; res < nMultires; res++)
            {
                int index = (myKeys[res]) << SMALL_BLOCK_SIZE_LOG2;
                mySHBins[index] = sumfunObj(mySHBins[index], myOut[res]);
            }
        }
    }
    else
    {
        OUTPUTTYPE myOut[nMultires];
        xformObj(input, myStart, &myKeys[0], &myOut[0], nMultires);
#pragma unroll
        for (int res = 0; res < nMultires; res++)
        {
            int index = (myKeys[res]) << SMALL_BLOCK_SIZE_LOG2;
            mySHBins[index] = sumfunObj(mySHBins[index], myOut[res]);
        }
    }
}
template <bool lastSteps, int nMultires,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
__global__
void histoKernel_smallBin(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero,
    OUTPUTTYPE* blockOut, int nOut, int maxblocks,
    int nSteps)
{
    // Take care with extern - In order to have two instances of this template the
    // type of the extern variables cannot change
    // (ie. cannot use "extern __shared__ OUTPUTTYPE bins[]")
    extern __shared__ int cudahistogram_allbinstmp[];
    OUTPUTTYPE* allbins = (OUTPUTTYPE*)&(*cudahistogram_allbinstmp);
    OUTPUTTYPE* mySHBins = &allbins[threadIdx.x];
    OUTPUTTYPE* ourOut = &blockOut[nOut * blockIdx.x];
    INDEXT myStart = start + (INDEXT)((blockIdx.x * nSteps) << SMALL_BLOCK_SIZE_LOG2) + (INDEXT)threadIdx.x;
    for (int bin = 0; bin < nOut /*- nLocVars*/; bin++)
        mySHBins[bin << SMALL_BLOCK_SIZE_LOG2] = zero;
    // Run loops - unroll 8 steps manually
    int doNSteps = (nSteps) >> 3;
    for (int step = 0; step < doNSteps; step++)
    {
        histoKernel_smallBinStep<lastSteps, nMultires>(input, xformObj, sumfunObj, myStart, end, mySHBins);
        histoKernel_smallBinStep<lastSteps, nMultires>(input, xformObj, sumfunObj, myStart + SMALL_BLOCK_SIZE, end, mySHBins);
        histoKernel_smallBinStep<lastSteps, nMultires>(input, xformObj, sumfunObj, myStart + 2*SMALL_BLOCK_SIZE, end, mySHBins);
        histoKernel_smallBinStep<lastSteps, nMultires>(input, xformObj, sumfunObj, myStart + 3*SMALL_BLOCK_SIZE, end, mySHBins);
        histoKernel_smallBinStep<lastSteps, nMultires>(input, xformObj, sumfunObj, myStart + 4*SMALL_BLOCK_SIZE, end, mySHBins);
        histoKernel_smallBinStep<lastSteps, nMultires>(input, xformObj, sumfunObj, myStart + 5*SMALL_BLOCK_SIZE, end, mySHBins);
        histoKernel_smallBinStep<lastSteps, nMultires>(input, xformObj, sumfunObj, myStart + 6*SMALL_BLOCK_SIZE, end, mySHBins);
        histoKernel_smallBinStep<lastSteps, nMultires>(input, xformObj, sumfunObj, myStart + 7*SMALL_BLOCK_SIZE, end, mySHBins);
        myStart += 8*SMALL_BLOCK_SIZE;
    }
    int nStepsLeft = (nSteps) - (doNSteps << 3);
    for (int step = 0; step < nStepsLeft; step++)
    {
        histoKernel_smallBinStep<true, nMultires>(input, xformObj, sumfunObj, myStart, end, mySHBins);
        myStart += SMALL_BLOCK_SIZE;
    }
    // In the end combine results:
#if SMALL_BLOCK_SIZE > 32
    __syncthreads();
#endif
    // Do first shared stuff:
    int keyIndex = threadIdx.x;
    while (keyIndex < nOut)
    {
        OUTPUTTYPE* binResults = &allbins[keyIndex << SMALL_BLOCK_SIZE_LOG2];
        OUTPUTTYPE result = ourOut[keyIndex];
        for (int tidx = 0; tidx < SMALL_BLOCK_SIZE; tidx++){
            result = sumfunObj(result, *binResults++);
        }
        ourOut[keyIndex] = result;
        keyIndex += SMALL_BLOCK_SIZE;
    }
}

static inline __device__
int resultToInt(int resultin){ return resultin; }
static inline __device__
int resultToInt(long resultin){ return (int)resultin; }
static inline __device__
int resultToInt(long long resultin){ return (int)resultin; }
static inline __device__
int resultToInt(unsigned int resultin){ return (int)resultin; }
static inline __device__
int resultToInt(unsigned long resultin){ return (int)resultin; }
static inline __device__
int resultToInt(unsigned long long resultin){ return (int)resultin; }

template<typename OUTPUTTYPE>
static inline __device__
int resultToInt(OUTPUTTYPE resultin){ return 0; }


static inline __device__
void intToResult(int resultin, int& resultOut){ resultOut = resultin; }
static inline __device__
void intToResult(int resultin, long& resultOut){ resultOut = (long)resultin; }
static inline __device__
void intToResult(int resultin, unsigned int& resultOut){ resultOut = (unsigned )resultin; }
static inline __device__
void intToResult(int resultin, long long& resultOut){ resultOut = (long long)resultin; }
static inline __device__
void intToResult(int resultin, unsigned long& resultOut){ resultOut = (unsigned long)resultin; }
static inline __device__
void intToResult(int resultin, unsigned long long& resultOut){ resultOut = (unsigned long long)resultin; }

template<typename OUTPUTTYPE>
static inline __device__
void intToResult(int resultin, OUTPUTTYPE& resultout){ ; }


template <bool lastSteps, int nMultires,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
static inline __device__
void histoKernel_smallBinByteOneStep(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT myStart, INDEXT end,
    volatile unsigned char* mySHBins,
    OUTPUTTYPE zero
    )
{
    if (lastSteps)
    {
        if (myStart < end)
        {
            OUTPUTTYPE myOut[nMultires];
            int myKeys[nMultires];
            xformObj(input, myStart, &myKeys[0], &myOut[0], nMultires);
#pragma unroll
            for (int res = 0; res < nMultires; res++)
            {
                // index = tid * 4 + (key / 4) * blockSize * 4 + (key % 4) - mySHBins points to allbins[4 x tid]
                // Complex indexing cost: 2x bit-shift + bitwise and + addition = 4 ops...
                int index = (((myKeys[res]) >> 2) << (SMALL_BLOCK_SIZE_LOG2 + 2)) + (myKeys[res] & 0x3);
                mySHBins[index]++;
            }
        }
    }
    else /*if (myStart < end)*/
    {
        OUTPUTTYPE myOut[nMultires];
        int myKeys[nMultires];
        xformObj(input, myStart, &myKeys[0], &myOut[0], nMultires);
#pragma unroll
        for (int res = 0; res < nMultires; res++)
        {
            // index = tid * 4 + (key / 4) * blockSize * 4 + (key % 4) - mySHBins points to allbins[4 x tid]
            // Complex indexing cost: 2x bit-shift + bitwise and + addition = 4 ops...
            int key = myKeys[res];
            int index = ((key >> 2) << (SMALL_BLOCK_SIZE_LOG2 + 2)) + (key & 0x3);
            mySHBins[index]++;
        }
    }
}




template <histogram_type histotype, bool lastSteps, int nMultires,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
__global__
void histoKernel_smallBinByte(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero,
    OUTPUTTYPE* blockOut, int nOut, int maxblocks,
    int nSteps)
{
    // Ok - idea is as follows: When we have blocksize number of threads, thread tid's nth-bin is at:
    //   index = tid * 4 + (bin / 4) * blocksize * 4 + (bin % 4)
    // Example:
    // With 32 threads bins #7, #8 and #9 will be at (7/4=1, 7%4=3, 8/4=2, 8%4=4, 9/4=2, 9%4=1):
    //     Bin #7       Bin #8      Bin #9   ...  Bin #63
    //  tid  | index    index       index    ...  index
    // ==============  ========    ========      ========
    //   0      35        256        257     ...  1923
    //   1      39        260        261     ...  1927
    //   2      43        264        265     ...  1931
    // ...
    //   31     255       380        381     ...  2047
    // Therefore there are blocksize x nOut number of 1-byte bins
    // Outputs are gathered from time to time to 32-bit bins
    //
    // Example2:
    // With 32 threads 7 bins
    //     Bin #0       Bin #1     Bin #2     Bin #3     Bin #4     Bin #5     Bin #6
    //  tid  | index    index      index      index      index      index      index
    // ==============  ========   ========   ========   ========   ========   ========
    //   0      0         1          2          3          128        129        130
    //   1      4         5          6          7          132        133        134
    //   2      8         9         10         11          136        137        138
    //  ...
    //   30     120      121        122        123         248        249        250
    //   31     124      125        126        127         252        253        254
    //
    // Example3:
    //   index = tid * 4 + (bin / 4) * blocksize * 4 + (bin % 4)
    // With 32 threads 3 bins
    //     Bin #0       Bin #1     Bin #2
    //  tid  | index    index      index
    // ==============  ========   ========
    //   0      0         1          2
    //   1      4         5          6
    //   2      8         9         10
    //  ...
    //   30     120      121        122
    //   31     124      125        126

    extern __shared__ unsigned char allbins2[];
    volatile unsigned char* mySHBins = &allbins2[threadIdx.x << 2];
    int padNOut = nOut + (((nOut & 0x3) != 0) ? (4 - (nOut & 0x3)) : 0);

    OUTPUTTYPE* ourOut = &blockOut[nOut * blockIdx.x];
#if __CUDA_ARCH__ >= 200
    OUTPUTTYPE* resultbins = ourOut;
#else
    OUTPUTTYPE* resultbins = (OUTPUTTYPE*)(&allbins2[padNOut << SMALL_BLOCK_SIZE_LOG2]);
#endif

    INDEXT myStart = start + (INDEXT)(((blockIdx.x * nSteps) << SMALL_BLOCK_SIZE_LOG2) + threadIdx.x);

    // Run loops
    //int nFullLoops = nSteps >> 7;
    // Clear bins
    {
        int* tmpSHBins = &((int*)allbins2)[threadIdx.x];
        // There are nOut x BLOCK_SIZE byte-sized bins so nOut x BLOCKISIZE/4 int-sized ones
        for (int bin = 0; bin < (padNOut >> 2) /*- nLocVars*/; bin++)
          tmpSHBins[bin << (SMALL_BLOCK_SIZE_LOG2)] = 0;
//        for (int tmpbin = (bin << 2); tmpbin < padNOut; tmpbin++)
//            mySHBins[tmpbin] = 0;
#if __CUDA_ARCH__ < 200
        int binid = threadIdx.x;
        while(binid < nOut)
        {
            resultbins[binid] = zero;
            binid += SMALL_BLOCK_SIZE;
        }
#endif
    }
#if SMALL_BLOCK_SIZE > 32
    __syncthreads();
#endif

    const int looplim = (255 / nMultires) < 63 ? (255 / nMultires) : 63;
    for (int stepsRem = nSteps; stepsRem > 0; stepsRem -= looplim)
    {
        if (stepsRem > looplim)
        {
#define MANUAL_UNROLL   1
#if MANUAL_UNROLL
            // Unroll manually
            // ("unexcpected control flow" construct with #pragma unroll)
#define DO_STEP(NUM) do { if ((NUM) < looplim) {                                \
    histoKernel_smallBinByteOneStep<lastSteps, nMultires>(                      \
        input, xformObj, sumfunObj, myStart /*+ (NUM) * SMALL_BLOCK_SIZE*/, end,\
        mySHBins, zero); myStart += SMALL_BLOCK_SIZE;                           \
    } } while (0)
#define DO_16_STEPS(N0)  do { \
    DO_STEP(N0 + 0); DO_STEP(N0 + 1); DO_STEP(N0 + 2); DO_STEP(N0 + 3);         \
    DO_STEP(N0 + 4); DO_STEP(N0 + 5); DO_STEP(N0 + 6); DO_STEP(N0 + 7);         \
    DO_STEP(N0 + 8); DO_STEP(N0 + 9); DO_STEP(N0 + 10); DO_STEP(N0 + 11);       \
    DO_STEP(N0 + 12); DO_STEP(N0 + 13); DO_STEP(N0 + 14); DO_STEP(N0 + 15);     \
    } while (0)

            DO_16_STEPS(0);
            DO_16_STEPS(16);
            DO_16_STEPS(32);
            DO_16_STEPS(48);
#undef      DO_16_STEPS
#undef      DO_STEP
            //myStart += looplim * SMALL_BLOCK_SIZE;
#else
            for (int stepNum = 0; stepNum < looplim; stepNum++){
              histoKernel_smallBinByteOneStep<lastSteps, nMultires>(
                input,
                xformObj,
                sumfunObj,
                myStart + stepNum * SMALL_BLOCK_SIZE, end,
                mySHBins, zero);
            }
            myStart += looplim * SMALL_BLOCK_SIZE;
#endif // MANUAL_UNROLL
#undef MANUAL_UNROLL
        }
        else
        {
            for (int stepNum = 0; stepNum < stepsRem; stepNum++){
              histoKernel_smallBinByteOneStep<lastSteps, nMultires>(
                input,
                xformObj,
                sumfunObj,
                myStart + stepNum * SMALL_BLOCK_SIZE, end,
                mySHBins, zero);
            }
            myStart += looplim * SMALL_BLOCK_SIZE;
        }
        // Ok  passes done - need to flush results together
        {
#       if SMALL_BLOCK_SIZE > 32
        __syncthreads();
#       endif
            int binid = threadIdx.x;
            while(binid < nOut)
            {
              // Start from own tid in order to avoid bank-conflicts:
              // index = tid * 4 + 4 * (bin / 4) * blocksize + (bin % 4)
              int index = (threadIdx.x << 2) + ((binid >> 2) << (SMALL_BLOCK_SIZE_LOG2 + 2)) + (binid & 0x3);
              //int res = (int)allbins2[index];
              int res = resultToInt(resultbins[binid]);
              int ilimit = SMALL_BLOCK_SIZE - threadIdx.x;
#pragma unroll
              for (int i=0; i < SMALL_BLOCK_SIZE; i++)
              {
                if (i == ilimit)
                  index -= (SMALL_BLOCK_SIZE << 2);
                res += allbins2[index];
                //allbins2[index] = 0;
                index += 4;
              }
              intToResult(res, resultbins[binid]);
              binid += SMALL_BLOCK_SIZE;
            }
#       if SMALL_BLOCK_SIZE > 32
        __syncthreads();
#       endif
            // zero the bins
            {
            int* tmpSHBins = &((int*)allbins2)[threadIdx.x];
            // There are nOut x BLOCK_SIZE byte-sized bins so nOut x BLOCKISIZE/4 int-sized ones
            for (int bin = 0; bin < (padNOut >> 2) /*- nLocVars*/; bin++)
              tmpSHBins[bin << (SMALL_BLOCK_SIZE_LOG2)] = 0;
            }

#       if SMALL_BLOCK_SIZE > 32
        __syncthreads();
#       endif
        }
    }
    // In the end combine results:
#if __CUDA_ARCH__ < 200
#if SMALL_BLOCK_SIZE > 32
    __syncthreads();
#endif
    int keyIndex = threadIdx.x;
    while (keyIndex < nOut)
    {
        OUTPUTTYPE result = ourOut[keyIndex];
        //result = result + resultbins[keyIndex];
        result = sumfunObj(result, *(OUTPUTTYPE*)(&resultbins[keyIndex]));
        ourOut[keyIndex] = result;
        keyIndex += SMALL_BLOCK_SIZE;
    }
#endif
}

template <histogram_type histotype, typename OUTPUTTYPE>
static int getSmallBinBufSize(int nOut, cudaDeviceProp* props)
{
    int maxblocks = props->multiProcessorCount * 3;

    maxblocks *= 2;
    if (nOut < 200) maxblocks *= 4;
    maxblocks *= 4;

    return (maxblocks + 1) * nOut * sizeof(OUTPUTTYPE);
}

template <histogram_type histotype, int nMultires,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
static
void callSmallBinHisto(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero,
    OUTPUTTYPE* out, int nOut,
    cudaDeviceProp* props,
    int cuda_arch,
    cudaStream_t stream,
    int* getTmpBufSize,
    void* tmpBuffer,
    bool outInDev)
{
    INDEXT size = end - start;
    if (end <= start)
    {
        if (getTmpBufSize) *getTmpBufSize = 0;
        return;
    }

    int maxblocks = props->multiProcessorCount * 3;


    if (size > 2*1024*1024 || getTmpBufSize){
        maxblocks *= 2;
        // High occupancy requires lots of blocks
        if (nOut < 200) maxblocks *= 4;
    }

    // TODO: Magic constants..
    // With low bin-counts and large problems it seems beneficial to use
    // more blocks...
    if (nOut <= 128 || size > 2*4096*4096 || getTmpBufSize)
        maxblocks *= 4;
    //printf("maxblocks = %d\n", maxblocks);

    OUTPUTTYPE* tmpOut;
    if (getTmpBufSize) {
        *getTmpBufSize = (maxblocks + 1) * nOut * sizeof(OUTPUTTYPE);
        return;
    }
    if (tmpBuffer)
        tmpOut = (OUTPUTTYPE*)tmpBuffer;
    else
        cudaMalloc((void**)&tmpOut, (maxblocks + 1) * nOut * sizeof(OUTPUTTYPE));
#if H_ERROR_CHECKS
    /*assert(getSmallBinBufSize<histotype, OUTPUTTYPE>(nOut, props) >=
            (maxblocks + 1) * nOut * sizeof(OUTPUTTYPE));*/
#endif
//    cudaMemset(tmpOut, 0, sizeof(OUTPUTTYPE) * nOut * (maxblocks+1));
    {
      #define IBLOCK_SIZE_LOG2    7
      #define IBLOCK_SIZE         (1 << IBLOCK_SIZE_LOG2)
      int initPaddedSize =
        ((maxblocks * nOut) + (IBLOCK_SIZE) - 1) & (~((IBLOCK_SIZE) - 1));
      const dim3 initblock = IBLOCK_SIZE;
      dim3 initgrid = initPaddedSize >> ( IBLOCK_SIZE_LOG2 );
      int nsteps = 1;
      while (initgrid.x > (1 << 14))
      {
          initgrid.x >>= 1;
          nsteps <<= 1;
          if (nsteps * initgrid.x * IBLOCK_SIZE < maxblocks * nOut)
              initgrid.x++;
      }
      initKernel<<<initgrid, initblock, 0, stream>>>(tmpOut, zero, maxblocks * nOut, nsteps);
      #undef IBLOCK_SIZE_LOG2
      #undef IBLOCK_SIZE
    }
    int sharedNeeded;
    if (histotype == histogram_atomic_inc)
    {
        int padNOut = nOut + (((nOut & 0x3) != 0) ? (4 - (nOut & 0x3)) : 0);
        sharedNeeded = (padNOut << SMALL_BLOCK_SIZE_LOG2);
        if (cuda_arch < 200)
            sharedNeeded += (nOut << 2);
    }
    else
    {
        int typesize = sizeof(OUTPUTTYPE);
        sharedNeeded = (nOut * typesize) << SMALL_BLOCK_SIZE_LOG2;
        //printf("Small-bin, generic, Shared needed = %d\n", sharedNeeded);
    }
    // Determine number of local variables
    // SMALL_LOCALLIMIT is total local size available for one block:

    int nSteps = size / (maxblocks << SMALL_BLOCK_SIZE_LOG2);
    if (nSteps * maxblocks * SMALL_BLOCK_SIZE < size) nSteps++;
    if (nSteps > MAX_SMALL_STEPS) nSteps = MAX_SMALL_STEPS;

    int nFullSteps = size / (nSteps * maxblocks * SMALL_BLOCK_SIZE);

    dim3 grid = maxblocks;
    dim3 block = SMALL_BLOCK_SIZE;
    for (int i = 0; i < nFullSteps; i++)
    {
        if (histotype == histogram_atomic_inc)
            histoKernel_smallBinByte<histotype, false, nMultires><<<grid, block, sharedNeeded, stream>>>(
                input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, maxblocks, nSteps);
        else
            histoKernel_smallBin<false, nMultires><<<grid, block, sharedNeeded, stream>>>(
                input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, maxblocks, nSteps);
        start += nSteps * maxblocks * SMALL_BLOCK_SIZE;
#if H_ERROR_CHECKS
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
           printf("Cudaerror = %s\n", cudaGetErrorString( error ));
#endif
    }
    size = end - start;
    if (size > 0)
    {
        // Do what steps we still can do without checks
        nSteps = size / (maxblocks << SMALL_BLOCK_SIZE_LOG2);
        if (nSteps > 0)
        {
            if (histotype == histogram_atomic_inc)
                histoKernel_smallBinByte<histotype, false, nMultires><<<grid, block, sharedNeeded, stream>>>(
                    input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, maxblocks, nSteps);
            else
                histoKernel_smallBin<false, nMultires><<<grid, block, sharedNeeded, stream>>>(
                    input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, maxblocks, nSteps);
            start += nSteps * maxblocks * SMALL_BLOCK_SIZE;
        }
    }
    size = end - start;
    if (size > 0)
    {
        // Last step here:
        int nblocks = size >> SMALL_BLOCK_SIZE_LOG2;
        if (nblocks >= maxblocks) nblocks = maxblocks;
        else if ((nblocks << SMALL_BLOCK_SIZE_LOG2) < size) nblocks++;
        nSteps = size / (nblocks << SMALL_BLOCK_SIZE_LOG2);
        if (nSteps * nblocks * SMALL_BLOCK_SIZE < size)
        {
            nSteps++;
            nblocks = size / (nSteps << SMALL_BLOCK_SIZE_LOG2);
            if (((nSteps * nblocks) << SMALL_BLOCK_SIZE_LOG2) < size) nblocks++;
        }
        grid.x = nblocks;
        if (histotype == histogram_atomic_inc)
            histoKernel_smallBinByte<histotype, true, nMultires><<<grid, block, sharedNeeded, stream>>>(
                input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, maxblocks, nSteps);
        else
            histoKernel_smallBin<true, nMultires><<<grid, block, sharedNeeded, stream>>>(
                input, xformObj, sumfunObj, start, end, zero, tmpOut, nOut, maxblocks, nSteps);
    }
#if H_ERROR_CHECKS
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
       printf("Cudaerror = %s\n", cudaGetErrorString( error ));
#endif
    // Finally put together the result:
    enum cudaMemcpyKind fromOut = outInDev ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    enum cudaMemcpyKind toOut = outInDev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
    if (stream != 0)
        cudaMemcpyAsync(&tmpOut[maxblocks * nOut], out, sizeof(OUTPUTTYPE) * nOut, fromOut, stream);
    else
        cudaMemcpy(&tmpOut[maxblocks * nOut], out, sizeof(OUTPUTTYPE) * nOut, fromOut);
    // Let's do so that one block handles one bin
    grid.x = nOut;
    //grid.x = nOut >> SMALL_BLOCK_SIZE_LOG2;
    //if ((grid.x << SMALL_BLOCK_SIZE_LOG2) < nOut) grid.x++;
    block.x = GATHER_BLOCK_SIZE;
    gatherKernel<<<grid, block, 0, stream>>>(sumfunObj, tmpOut, nOut, maxblocks, zero);
    // TODO: Use async copy for the results as well?
    if (outInDev && stream != 0)
        cudaMemcpyAsync(out, tmpOut, nOut*sizeof(OUTPUTTYPE), toOut, stream);
    else
        cudaMemcpy(out, tmpOut, nOut*sizeof(OUTPUTTYPE), toOut);
    #if 0
    {
      int resIdx;
      int i;
      OUTPUTTYPE* h_tmp = (OUTPUTTYPE*)malloc(maxblocks * nOut * sizeof(OUTPUTTYPE));
      //parallel_copy(h_tmp, MemType_HOST, tmpOut, MemType_DEV, n * nOut * sizeof(OUTPUTTYPE));
      cudaMemcpy(h_tmp, tmpOut, maxblocks*nOut*sizeof(OUTPUTTYPE), cudaMemcpyDeviceToHost);
      for (resIdx = 0; resIdx < nOut; resIdx++)
      {
        OUTPUTTYPE res = out[resIdx];
        for (i = 0; i < maxblocks; i++)
        {
          res = sumfunObj(res, h_tmp[i * nOut + resIdx]);
        }
        out[resIdx] = sumfunObj(res, out[resIdx]);
      }
      free(h_tmp);
    }
    #endif
    if (!tmpBuffer)
        cudaFree(tmpOut);
}

template <histogram_type histotype, typename OUTPUTTYPE>
static inline
bool smallBinLimit(int nOut, OUTPUTTYPE zero, cudaDeviceProp* props, int cuda_arch)
{
  int shlimit = props->sharedMemPerBlock - 300;

  int typeSize = sizeof(OUTPUTTYPE);

  if (histotype == histogram_atomic_inc)
      if ((((4 * nOut) << 5) + (cuda_arch < 200 ? nOut * 16 : 0)) < shlimit)
          return true;

  if (((4 * nOut * typeSize) << 5) < shlimit)
      return true;
  return false;
}

__global__
void detectCudaArchKernel(int* res)
{
    int result;
#if __CUDA_ARCH__ >= 210
    result = 210;
#elif __CUDA_ARCH__ >= 200
    result = 200;
#elif __CUDA_ARCH__ >= 130
    result = 130;
#elif __CUDA_ARCH__ >= 120
    result = 120;
#elif __CUDA_ARCH__ >= 110
    result = 110;
#else
    result = 100;
#endif
    if (threadIdx.x == 0)
        *res = result;
}

static
int DetectCudaArch(void)
{
    // The only way to know from host-code, which device architecture our kernels have been generated
    // against, is to run a kernel that actually checks it.. :)
    dim3 grid = 1;
    //dim3 block = 32;
    // TODO: Allow static storage so that we can ask just once for the arch???
    // NOTE: This function implies synchromization between CPU and GPU - so use static here...
    static int result = 0;
    //int result = 0;
    if (result == 0)
    {
        void* tmpBuf;
        cudaMalloc(&tmpBuf, sizeof(int));
        detectCudaArchKernel<<<grid, grid>>>((int*)tmpBuf);
        cudaMemcpy(&result, tmpBuf, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(tmpBuf);
      //printf("Detected CUDA_ARCH = %d\n", result);
    }
    return result;
}

static bool runMultiPass(int nOut, cudaDeviceProp* props, int cuda_arch, size_t outsize, histogram_type histotype)
{
    int subsize = determineSubHistoSize(nOut, outsize, histotype, cuda_arch, props);

    if (cuda_arch < 120){
        if (subsize <= 0 || nOut > 2 * subsize)
            return false;
        return true;
    }
    else
    {
        if (subsize <= 0 || nOut > 16 * subsize)
            return false;
        return true;
    }
}


template <histogram_type histotype, int nMultires,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
cudaError_t
callHistogramKernel(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT start, INDEXT end,
    OUTPUTTYPE zero, OUTPUTTYPE* out, int nOut,
    bool outInDev,
    cudaStream_t stream, void* tmpBuffer,
    bool allowMultiPass)
{

    int devId;
    cudaDeviceProp props;
    cudaError_t cudaErr = cudaGetDevice( &devId );
    if (cudaErr != 0) return cudaErr;
    //assert(!cudaErr);
    cudaErr = cudaGetDeviceProperties( &props, devId );
    if (cudaErr != 0) return cudaErr;
    int cuda_arch = DetectCudaArch();

    enum cudaFuncCache old;
    cudaThreadGetCacheConfig(&old);
    cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

    if (nOut <= 0) return cudaSuccess;
    // 100 Mib printf-limit should be enough...
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 100);

    if (smallBinLimit<histotype>(nOut, zero, &props, cuda_arch))
    {
        callSmallBinHisto<histotype, nMultires>(input, xformObj, sumfunObj, start, end, zero, out, nOut, &props, cuda_arch, stream, NULL, tmpBuffer, outInDev);
    }
    else if (binsFitIntoShared(nOut, zero, &props, cuda_arch))
    {
        callHistogramKernelImpl<histotype, nMultires>(input, xformObj, sumfunObj, start, end, zero, out, nOut,  &props, stream, NULL, tmpBuffer, outInDev, cuda_arch);
    }
    else if (allowMultiPass && runMultiPass(nOut, &props, cuda_arch, sizeof(OUTPUTTYPE), histotype))
    {
        callHistogramKernelMultiPass<histotype, nMultires>(input, xformObj, sumfunObj, start, end, zero, out, nOut,  &props, stream, tmpBuffer, outInDev, cuda_arch);
    }
    else
    {
        callHistogramKernelLargeNBins<histotype, nMultires>(input, xformObj, sumfunObj, start, end, zero, out, nOut, &props, cuda_arch, stream, NULL, tmpBuffer, outInDev);
    }
    cudaThreadSetCacheConfig(old);
    return cudaSuccess;
}

template <typename nDimIndexFun, int nDim, typename USERINPUTTYPE, typename INDEXT, typename OUTPUTTYPE>
class wrapHistoInput
{
public:
    nDimIndexFun userIndexFun;
    INDEXT starts[nDim];
    //int ends[nDim];
    INDEXT sizes[nDim];
    __host__ __device__
    void operator() (USERINPUTTYPE input, INDEXT i, int* result_index, OUTPUTTYPE* results, int nresults) const {
        int coords[nDim];
        int tmpi = i;
  #pragma unroll
        for (int d=0; d < nDim; d++)
        {
            // Example of how this logic works - imagine a cube of (10,100,1000), and take index 123 456
            // newI = 123 456 / 10 = 12 345, offset = 123 456 - 123 450 = 6 (this is our first coordinate!),
            // newI = 12 345 / 100 = 123,    offset = 12 345 - 12 300 = 45 (this is our second coordinate!),
            // newI = 123 / 1000 = 0,        offset = 123 - 0 = 123 (this is our last coordinate!)
            // Result = [123, 45, 6]
            INDEXT newI = tmpi / sizes[d];
            INDEXT offset = tmpi - newI * sizes[d];
            coords[d] = starts[d] + offset;
            tmpi = newI;
        }
        // Now just call wrapped functor with right coordinate values
        userIndexFun(input, coords, result_index, results, nresults);
    }
};



template <histogram_type histotype, int nMultires, int nDim,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
cudaError_t
callHistogramKernelNDim(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT* starts, INDEXT* ends,
    OUTPUTTYPE zero, OUTPUTTYPE* out, int nOut,
    bool outInDev,
    cudaStream_t stream, void* tmpBuffer,
    bool allowMultiPass)
{
    wrapHistoInput<TRANSFORMFUNTYPE, nDim, INPUTTYPE, INDEXT, OUTPUTTYPE> wrapInput;
    INDEXT start = 0;
    INDEXT size = 1;
    for (int d = 0; d < nDim; d++)
    {
        wrapInput.starts[d] = starts[d];
        wrapInput.sizes[d] = ends[d] - starts[d];
        // Example: starts = [3, 10, 23], sizes = [10, 100, 1000]
        // start = 3 * 1 = 3, size = 10
        // start = 3 + 10 * 10 = 103, size = 10*100 = 1000
        // start = 103 + 1000*23 = 23 103, size = 1000*1000 = 1 000 000
        start += starts[d] * size;
        size *= wrapInput.sizes[d];
        if (ends[d] <= starts[d]) return cudaSuccess;
    }
    wrapInput.userIndexFun = xformObj;
    INDEXT end = start + size;
    return callHistogramKernel<histotype, nMultires>
        (input, wrapInput, sumfunObj, start, end, zero, out, nOut, outInDev, stream, tmpBuffer, allowMultiPass);
}

template <histogram_type histotype, int nMultires,
    typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename SUMFUNTYPE,
    typename OUTPUTTYPE, typename INDEXT>
cudaError_t
callHistogramKernel2Dim(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    SUMFUNTYPE sumfunObj,
    INDEXT x0, INDEXT x1,
    INDEXT y0, INDEXT y1,
    OUTPUTTYPE zero, OUTPUTTYPE* out, int nOut,
    bool outInDev,
    cudaStream_t stream, void* tmpBuffer,
    bool allowMultiPass)
{
    INDEXT starts[2] = { x0, y0 };
    INDEXT ends[2] = { x1, y1 };
    return callHistogramKernelNDim<histotype, nMultires, 2>
        (input, xformObj, sumfunObj, starts, ends, zero, out, nOut, outInDev, stream, tmpBuffer, allowMultiPass);

}


struct histogram_defaultXform
{
  __host__ __device__
  void operator() (int* input, int i, int* result_index, int* results, int nresults) const {
      //int idata = input[i];
#pragma unroll
      for (int resIndex = 0; resIndex < nresults; resIndex++)
      {
        *result_index++ = *input++;
        *results++ = 1;
      }
  }
};


template <typename OUTPUTTYPE>
struct histogram_defaultSum
{
    __host__ __device__
    OUTPUTTYPE operator() (OUTPUTTYPE i1, OUTPUTTYPE i2) const {
        return i1 + i2;
    }
};

template <typename INPUTTYPE, typename OUTPUTTYPE>
struct histogram_dummyXform
{
  __host__ __device__
  void operator() (INPUTTYPE* input, int i, int* result_index, OUTPUTTYPE* results, int nresults) const {
      //int idata = input[i];
      int index = i;
      (void)input;
#pragma unroll
      for (int resIndex = 0; resIndex < nresults; resIndex++)
      {
        *result_index++ = index++;
        *results++ = 1;//*input++;
      }
  }
};


template <typename OUTPUTTYPE>
struct histogram_dummySum
{
    __host__ __device__
    OUTPUTTYPE operator() (OUTPUTTYPE i1, OUTPUTTYPE i2) const {
        return i1;
    }
};


template <histogram_type histotype, typename OUTPUTTYPE>
int getHistogramBufSize(OUTPUTTYPE zero, int nOut)
{
    int result = 0;
    int devId;
    cudaDeviceProp props;
    cudaError_t cudaErr = cudaGetDevice( &devId );
    if (cudaErr != 0) return -1;
    //assert(!cudaErr);
    cudaErr = cudaGetDeviceProperties( &props, devId );
    if (cudaErr != 0) return -1;
    int cuda_arch = DetectCudaArch();
    if (nOut <= 0) return 0;
    if (smallBinLimit<histotype>(nOut, zero, &props, cuda_arch))
    {
        result = getSmallBinBufSize<histotype, OUTPUTTYPE>(nOut, &props);
    }
    else if (binsFitIntoShared(nOut, zero, &props, cuda_arch))
    {
        result = getMediumHistoTmpbufSize<histotype, OUTPUTTYPE>(nOut, &props);
    }
    else if (runMultiPass(nOut, &props, cuda_arch, sizeof(OUTPUTTYPE), histotype))
    {
        result = getMultipassBufSize<histotype, OUTPUTTYPE>(nOut, &props, cuda_arch);
    }
    else
    {
        result = getLargeBinTmpbufsize<histotype, OUTPUTTYPE>(nOut, &props, cuda_arch);
    }
    return result;
}


// undef everything

#undef H_ERROR_CHECKS
#undef HBLOCK_SIZE_LOG2
#undef HBLOCK_SIZE
#undef HMBLOCK_SIZE_LOG2
#undef HMBLOCK_SIZE
#undef LBLOCK_SIZE_LOG2
#undef LBLOCK_SIZE
#undef GATHER_BLOCK_SIZE_LOG2
#undef GATHER_BLOCK_SIZE
#undef LBLOCK_WARPS
#undef RBLOCK_SIZE
#undef RMAXSTEPS
#undef NHSTEPSPERKEY
#undef MAX_NHSTEPS
#undef MAX_MULTISTEPS
#undef MAX_NLHSTEPS
#undef STRATEGY_CHECK_INTERVAL_LOG2
#undef STRATEGY_CHECK_INTERVAL
#undef HASH_COLLISION_STEPS
#undef USE_JENKINS_HASH
#undef LARGE_NBIN_CHECK_INTERVAL_LOG2
#undef LARGE_NBIN_CHECK_INTERVAL
#undef SMALL_BLOCK_SIZE_LOG2
#undef SMALL_BLOCK_SIZE
#undef MAX_SMALL_STEPS
#undef USE_ATOMICS_HASH
#undef USE_BALLOT_HISTOGRAM
#undef TAKE_WARP_MUTEX
#undef GIVE_WARP_MUTEX
#undef FREE_MUTEX_ID


#if USE_MEDIUM_PATH
#undef MEDIUM_BLOCK_SIZE_LOG2
#undef MEDIUM_BLOCK_SIZE
#endif
#undef USE_MEDIUM_PATH




#endif /* CUDA_HISTOGRAM_H_ */
