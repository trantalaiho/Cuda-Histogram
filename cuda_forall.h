/*
 * cuda_forall.h
 *
 *  Created on: 27.3.2012
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

#include <cuda_runtime_api.h>
#include "cuda_rnd.h"

#define FOR_ALL_BLOCK_SIZE_LOG2     6
#define FOR_ALL_BLOCK_SIZE          (1 << FOR_ALL_BLOCK_SIZE_LOG2)

// As simple as possible

template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE>
__global__
void cuda_transFormKernel(INPUTTYPE input, TRANSFORMFUNTYPE functor, INDEXTYPE start, INDEXTYPE end, int nsteps) {
    INDEXTYPE idx = (INDEXTYPE)(blockIdx.x * blockDim.x + threadIdx.x) + start;
    for (int step = 0; step < nsteps; step++){
        if (idx < end){
          functor(input, idx, blockIdx.y);
        }
        idx += blockDim.x * gridDim.x;
    }
}


template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE>
static inline
void callTransformKernel(
    INPUTTYPE input,
    TRANSFORMFUNTYPE functionObject,
    INDEXTYPE start, INDEXTYPE end,
    int nMultiXform = 1,
    cudaStream_t stream = 0) {
  if (end <= start)
    return;
  INDEXTYPE size = end - start;
  const dim3 block = FOR_ALL_BLOCK_SIZE;
  int paddedSize = (size + (FOR_ALL_BLOCK_SIZE) - 1) & (~((FOR_ALL_BLOCK_SIZE) - 1));
  dim3 grid = paddedSize >> ( FOR_ALL_BLOCK_SIZE_LOG2 );
  grid.y = nMultiXform;
  int steps = 1;
  if (grid.x > (1 << 12)){
      grid.x = (1 << 12);
      steps = size >> (FOR_ALL_BLOCK_SIZE_LOG2 + 12);
      if (steps << (FOR_ALL_BLOCK_SIZE_LOG2 + 12) < size) steps++;
  }
  cuda_transFormKernel<<<grid, block, 0, stream>>>(input, functionObject, start, end, steps);
}

// TODO: Error reporting?

// TODO: WHEN CALLING GENERATE MUST NOT BE FROM A WARP THAT HAS INTRA-WARP DIVERGENCIES!

#define MAX_BLOCKS_LOG2     10
#define MAX_BLOCKS          (1 << MAX_BLOCKS_LOG2)


#define GENERATE_RND(X) GENERATE_WS_RND()

template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE>
__global__
void cuda_transFormKernelRnds(INPUTTYPE input, TRANSFORMFUNTYPE functor, INDEXTYPE start, INDEXTYPE end, unsigned int* rnd_state, int nsteps)
{
  INIT_WS_STATE(rnd_state, FOR_ALL_BLOCK_SIZE);
  INDEXTYPE idx = (INDEXTYPE)(blockIdx.x * blockDim.x + threadIdx.x) + start;
  for (int step = 0; step < nsteps; step++){
//      if (idx < end){
        functor(input, idx);
//      }
      idx += blockDim.x *gridDim.x;
  }
  if (idx < end){
    SET_SAFE_MODE(true);
  }
  while (idx < end){
      functor(input, idx);
      idx += blockDim.x *gridDim.x;
  }
  SAVE_WS_STATE(rnd_state);
}





template <typename TRANSFORMFUNTYPE, typename INDEXTYPE, typename INPUTTYPE>
static inline
void callxFormKernel_UseRnds(
    INPUTTYPE input,
    TRANSFORMFUNTYPE functionObject,
    INDEXTYPE start, INDEXTYPE end,
    cudaStream_t stream = 0) {
  if (end <= start)
    return;
  static int rnd_state_size = 0;
  static unsigned int* rnd_state = NULL;
  INDEXTYPE size = end - start;
  const dim3 block = FOR_ALL_BLOCK_SIZE;
  int paddedSize = (size + (FOR_ALL_BLOCK_SIZE) - 1) & (~((FOR_ALL_BLOCK_SIZE) - 1));
  dim3 grid = paddedSize >> ( FOR_ALL_BLOCK_SIZE_LOG2 );
  int steps = 1;
  if (grid.x > MAX_BLOCKS){
      grid.x = MAX_BLOCKS;
      steps = size >> (FOR_ALL_BLOCK_SIZE_LOG2 + MAX_BLOCKS_LOG2);
      if (steps << (FOR_ALL_BLOCK_SIZE_LOG2 + MAX_BLOCKS_LOG2) < size) steps++;
  }
  size_t rndSize = grid.x << FOR_ALL_BLOCK_SIZE_LOG2;
  if (rnd_state_size < rndSize){
      if (rnd_state) cudaFree(rnd_state);
      rnd_state = seedRnds(true, rndSize);
      if (!rnd_state) rnd_state = seedRnds(false, rndSize);
      rnd_state_size = rndSize;
      if (!rnd_state)
          return;
  }
  cuda_transFormKernelRnds<<<grid, block, 0, stream>>>(input, functionObject, start, end, rnd_state, steps);
  start += steps << (FOR_ALL_BLOCK_SIZE_LOG2 + MAX_BLOCKS_LOG2);

}







template <typename nDimIndexFun, int nDim, typename USERINPUTTYPE, typename INDEXT>
class wrapXFormInput
{
public:
    nDimIndexFun userIndexFun;
    INDEXT starts[nDim];
    //int ends[nDim];
    INDEXT sizes[nDim];
    __host__ __device__
    void operator() (USERINPUTTYPE input, INDEXT i, int multiIndex) const {
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
        userIndexFun(input, coords, multiIndex);
    }
};



template <int nDim, typename INPUTTYPE, typename TRANSFORMFUNTYPE, typename INDEXT>
cudaError_t
callXformKernelNDim(
    INPUTTYPE input,
    TRANSFORMFUNTYPE xformObj,
    INDEXT* starts, INDEXT* ends,
    int nMultiXform = 1,
    cudaStream_t stream = 0)
{
    wrapXFormInput<TRANSFORMFUNTYPE, nDim, INPUTTYPE, INDEXT> wrapInput;
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

    callTransformKernel(input, wrapInput, start, end, nMultiXform, stream);
    return cudaSuccess;
}



#undef FOR_ALL_BLOCK_SIZE
#undef FOR_ALL_BLOCK_SIZE_LOG2



