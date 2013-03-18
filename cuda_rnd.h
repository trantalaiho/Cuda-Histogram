/*
 * cuda_rnd.h
 *
 *  Created on: 31.3.2012
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
 *  Uses WarpStandard Algorithm by David B. Thomas (see warp_standard.h)
 *
 */

#ifndef CUDA_RND_H_
#define CUDA_RND_H_

#include "warp_standard.h"
#include <cuda_runtime_api.h>
#include <stdio.h>
// Instructions: just use the three macros below and get good online random numbers
// Pre-generating

// Note NTHREADS_PER_BLOCK has to be compile-time constant (and multiple of 32) for
// this to work!

// TODO: Experimental CODE! Quality of random numbers not yet tested, so expect bugs here!

__shared__ unsigned int * s_warp_standard_shmem;
__device__ unsigned int * s_warp_standard_regs;
// Turn this flag on if you have divergent threads within a warp! DEFAULT: OFF
__shared__ bool s_warp_safe_mode;

#define INIT_WS_STATE(STATEPTR, NTHREADS_PER_BLOCK)                             \
    unsigned int            tmpregs[WarpStandard_REG_COUNT];                    \
    __shared__ unsigned int tmpshs[NTHREADS_PER_BLOCK];                         \
    WarpStandard_LoadState(STATEPTR, tmpregs, tmpshs);                          \
    s_warp_standard_shmem = tmpshs;                                             \
    s_warp_safe_mode = false;                                                   \
    s_warp_standard_regs = tmpregs


#ifndef __CUDA_ARCH__
#define GENERATE_WS_RND(X)  0
#define SET_SAFE_MODE(FLAG)
#else
#define GENERATE_WS_RND(X) WarpStandard_Generate(s_warp_standard_regs, s_warp_standard_shmem, s_warp_safe_mode)
#define SET_SAFE_MODE(FLAG) s_warp_safe_mode = (FLAG);
#endif

#define SAVE_WS_STATE(STATEPTR) WarpStandard_SaveState(s_warp_standard_regs, s_warp_standard_shmem, STATEPTR)

// Alternatively just use the functions:


// uses either /dev/urandom or lrand48() (Latter will have worse quality)
static unsigned int* seedRnds(bool useDevRandom, int nThreadsTot);

unsigned int* seedRnds(bool useDevRandom, int nThreadsTot)
{
    unsigned int* result;
    unsigned int* h_result;
    cudaMalloc(&result, nThreadsTot * sizeof(unsigned int));
    h_result = (unsigned int*)malloc(nThreadsTot * sizeof(unsigned int));
    if (!result || !h_result)
        return result;
    if (useDevRandom){
        FILE* rndf =  fopen("/dev/urandom", "r");
        for (int i = 0; i < nThreadsTot; i++)
            if (fread(&h_result[i], sizeof(unsigned int), 1, rndf) != 1){
              fclose(rndf);
              cudaFree(result);
              return NULL;
            }
        fclose(rndf);
    } else {
        for (int i = 0; i < nThreadsTot; i++){
            unsigned int rnd1 = (unsigned int)lrand48();
            unsigned int rnd2 = (unsigned int)lrand48();
            h_result[i] = ((rnd1 & 0x00FFFF00u) << 8) | ((rnd2 & 0x00FFFF00u) >> 8);
        }
    }
    cudaMemcpy(result, h_result, nThreadsTot * sizeof(unsigned int), cudaMemcpyHostToDevice);
    free(h_result);
    return result;
}

#define GEN_RND_BLOCKSIZE_LOG2      7
#define GEN_RND_BLOCKSIZE           (1 << GEN_RND_BLOCKSIZE_LOG2)
#define GEN_RND_MAXBLOCKS_LOG2      6
#define GEN_RND_MAXBLOCKS           (1 << GEN_RND_MAXBLOCKS_LOG2)



__global__
void cuda_rndgenKernel(unsigned int* output, unsigned int* state, size_t nOutputs ,int nsteps) {
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int            tmpregs[WarpStandard_REG_COUNT];
    __shared__ unsigned int tmpshs[GEN_RND_BLOCKSIZE];
    WarpStandard_LoadState(state, tmpregs, tmpshs);
    for (int step = 0; step < nsteps; step++){
        unsigned int rndout = WarpStandard_Generate(tmpregs, tmpshs, false);
        if (idx < nOutputs){
            output[idx] = rndout;
        }
        idx += blockDim.x * gridDim.x;
    }
    WarpStandard_SaveState(tmpregs, tmpshs, state);
}

static unsigned int* cudaGenerateRnds(
    size_t nOutputs,
    unsigned int* result = NULL,
    bool newseed = false,
    bool useDevRndSeed = true,
    int seed = 0,
    cudaStream_t stream = 0)
{
    static unsigned int* state = NULL;
    if (!state || newseed){
        if (state) cudaFree(state);
        state = seedRnds(useDevRndSeed, GEN_RND_BLOCKSIZE * GEN_RND_MAXBLOCKS);
        if (!state) return NULL;
    }
    if (nOutputs <= 0)
      return NULL;
    const dim3 block = GEN_RND_BLOCKSIZE;
    dim3 grid = nOutputs >> ( GEN_RND_BLOCKSIZE_LOG2 );
    int steps = 1;
    if (grid.x > GEN_RND_MAXBLOCKS){
        grid.x = GEN_RND_MAXBLOCKS;
        steps = nOutputs >> (GEN_RND_BLOCKSIZE_LOG2 + GEN_RND_MAXBLOCKS_LOG2);
        if (steps << (GEN_RND_BLOCKSIZE_LOG2 + GEN_RND_MAXBLOCKS_LOG2) < nOutputs) steps++;
    }
    if (!result){
        cudaMalloc(&result, sizeof(unsigned int) * nOutputs);
        if (!result) return result;
    }
    cuda_rndgenKernel<<<grid, block, 0, stream>>>(result, state, nOutputs, steps);
    return result;
}


#endif /* CUDA_RND_H_ */
