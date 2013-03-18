/*
 * warp_standard.h
 *
 *  Created on: Mar 31, 2012
 *      Implements WarpStandard Algorithm by David B. Thomas
 *      See: http://www.doc.ic.ac.uk/~dt10/research/rngs-gpu-warp_generator.html
 *      Author claims BSD-license -- see below, or http://www.linfo.org/bsdlicense.html
 *
 *
 *
 *   Copyright © 2011-2012 David B. Thomas. All Rights Reserved.
 *
 *   Redistribution and use in source and binary forms, with or without modification,
 *   are permitted provided that the following conditions are met:
 *
 *    1. Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright notice,
 *        this list of conditions and the following disclaimer in the documentation
 *        and/or other materials provided with the distribution.
 *
 *    3. The name of the author may not be used to endorse or promote products derived
 *        from this software without specific prior written permission.
 *
 *    THIS SOFTWARE IS PROVIDED BY [LICENSOR] "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
 *    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 *    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *    EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *
 *
 */


// My NOTES:

/*
 *
 * - Algorithm uses one word per thread of shared memory
 * - State-size is number of total warps multiplied by 32 -> Number of threads words
 * - As you can see below, it uses 3 registers that need to be allocated for it
 * - The state has to be initialized to good random numbers if you want good results
 *      (meaning author recommends usage of /dev/urandom)
 */



#ifndef WARP_STANDARD_H_
#define WARP_STANDARD_H_

#include <cuda_runtime_api.h>

__device__ void WarpStandard_LoadState(const unsigned *seed, unsigned *regs, unsigned *shmem);
__device__ void WarpStandard_SaveState(const unsigned *regs, const unsigned *shmem, unsigned *seed);
__device__ unsigned WarpStandard_Generate(unsigned *regs, unsigned *shmem, bool safeMode);


/////////////////////////////////////////////////////////////////////////////////////
// Public constants

const unsigned WarpStandard_K=32;
const unsigned WarpStandard_REG_COUNT=3;
const unsigned WarpStandard_STATE_WORDS=32;

const unsigned WarpStandard_TEST_DATA[WarpStandard_STATE_WORDS]={
    0x8cf35fea, 0xe1dd819e, 0x4a7d0a8e, 0xe0c05911, 0xfd053b8d, 0x30643089, 0x6f6ac111, 0xc4869595, 0x9416b7be, 0xe6d329e8, 0x5af0f5bf, 0xc5c742b5, 0x7197e922, 0x71aa35b4, 0x2070b9d1, 0x2bb34804, 0x7754a517, 0xe725315e, 0x7f9dd497, 0x043b58bf, 0x83ffa33d, 0x2532905a, 0xbdfe0c8a, 0x16f68671, 0x0d14da2e, 0x847efd5f, 0x1edeec64, 0x1bebdf9b, 0xf74d4ff3, 0xd404774b, 0x8ee32599, 0xefe0c405
};

//////////////////////////////////////////////////////////////////////////////////////
// Private constants

const char *WarpStandard_name="WarpRNG[CorrelatedU32Rng;k=32;g=16;rs=0;w=32;n=1024;hash=deac2e12ec6e615]";
const char *WarpStandard_post_processing="addtaps";
const unsigned WarpStandard_N=1024;
const unsigned WarpStandard_W=32;
const unsigned WarpStandard_G=16;
const unsigned WarpStandard_SR=0;
__device__ const unsigned WarpStandard_Q[2][32]={
  {29,24,5,23,14,26,11,31,9,3,1,28,0,2,22,20,18,15,27,13,10,16,8,17,25,12,19,30,7,6,4,21},
  {5,14,28,24,19,13,0,17,11,20,7,10,6,15,2,9,8,23,4,30,12,25,3,21,26,27,31,18,22,16,29,1}
};
const unsigned WarpStandard_Z0=2;
__device__ const unsigned WarpStandard_Z1[32]={
  0,1,0,1,1,1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1};

const unsigned WarpStandard_SHMEM_WORDS=32;
const unsigned WarpStandard_GMEM_WORDS=0;

////////////////////////////////////////////////////////////////////////////////////////
// Public functions

void WarpStandard_LoadState(const unsigned *seed, unsigned *regs, unsigned *shmem)
{
  unsigned offset=threadIdx.x % 32;  unsigned base=threadIdx.x-offset;
  // setup constants
  regs[0]=WarpStandard_Z1[offset];
  regs[1]=base + WarpStandard_Q[0][offset];
  regs[2]=base + WarpStandard_Q[1][offset];
  // Setup state
  unsigned stateOff=blockDim.x * blockIdx.x * 1 + threadIdx.x * 1;
  shmem[threadIdx.x]=seed[stateOff];
}

void WarpStandard_SaveState(const unsigned *regs, const unsigned *shmem, unsigned *seed)
{
  unsigned stateOff=blockDim.x * blockIdx.x * 1 + threadIdx.x * 1;
  seed[stateOff] = shmem[threadIdx.x];
}

unsigned WarpStandard_Generate(unsigned *regs, unsigned *shmem, bool safeMode)
{
  if (!safeMode){
#if __DEVICE_EMULATION__
    __syncthreads();
#endif
  unsigned t0=shmem[regs[1]], t1=shmem[regs[2]];
  unsigned res=(t0<<WarpStandard_Z0) ^ (t1>>regs[0]);
#if __DEVICE_EMULATION__
    __syncthreads();
#endif
  shmem[threadIdx.x]=res;
  return t0+t1;
  } else {
    // TODO Can this work? :) Hacky...
    // Find one thread that is alive
    __shared__ volatile int aliveId;
    aliveId = threadIdx.x;
    //__threadsync_block();
    unsigned myres = 0;
    for (int tid = 0; tid < blockDim.x; tid++){
        unsigned res = 0;
        if (aliveId == threadIdx.x){
            unsigned offset=tid % 32;  unsigned base=tid-offset;
            unsigned regs[3];
            // setup constants
            regs[0]=WarpStandard_Z1[offset];
            regs[1]=base + WarpStandard_Q[0][offset];
            regs[2]=base + WarpStandard_Q[1][offset];

            unsigned t0=shmem[regs[1]], t1=shmem[regs[2]];
            unsigned res=(t0<<WarpStandard_Z0) ^ (t1>>regs[0]);
            shmem[tid]=t0+t1;
        }
        // TODO How to ensure that write is visible with older CUDA-versions?
        // Within a warp this should work always correctly, but between warps what happens?
        //__threadsync_block();
        if (threadIdx.x == tid)
            myres = shmem[tid];
        if (aliveId == threadIdx.x)
            shmem[tid]=res;
    }
    return myres;
  }
};






#endif /* WARP_STANDARD_H_ */
