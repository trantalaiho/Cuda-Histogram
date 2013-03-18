/*
 *
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
 *
 */


// Compile for CUDA with:    nvcc -x cu -O4 -arch=<your_arch> -I../ ising.cpp -DCUDA -o ising_gpu
// and for CPU:              g++ -O4 ising.cpp -lm -o ising_cpu

// Optionally include -DGAUGE to compile the Ising Gauge model simulator


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mersenne.h"



#ifdef CUDA
#define radix   float
#else
#define radix   float
#endif


// Our lattice-field - Ising model: one bit per spin:
typedef unsigned int uint32;



// Define fields that are present in simulation:
#ifdef GAUGE
#define ISING_GAUGE 1
#define ISING_SITES 0
#else
#define ISING_GAUGE 0
#define ISING_SITES 1
#endif

// Define system - now length of array dims defines number of dims..
static const int DIMS[] = { 256, 256 };
static const radix s_beta = 0.8;
#define NHITS   10
#define DEFAULT_TRAJS   40000
#define NTHERMAL        500

#define HOT_START       0



// Below are some helper defines - debug-flags and so forth

#define LEN_OF_ARRAY(X) (sizeof(X)/sizeof(X[0]))
#define NDIM LEN_OF_ARRAY(DIMS)

typedef struct updateInput_s
{
    radix beta;
#if ISING_GAUGE
    int dir;
    uint32* spinLinks[NDIM];
#endif
#if ISING_SITES
    uint32* spins;
#endif
} updateInput;



#define USE_NEW_RND     1
#define PREGEN_NEW_RND  0


#ifdef CUDA
#define USE_CUDA    1
#else
#define USE_CUDA    0
#endif

#define DEBUG_NB_ARRAYS 0
#define DEBUG_CUDA  0


#ifndef CUDA
#undef DEBUG_CUDA
#define DEBUG_CUDA  0
#endif

#if DEBUG_CUDA
  #define CHECK_CUDA_ERROR(X) do { \
      cudaError_t error = cudaGetLastError(); \
        if (error){ \
            printf("%s, line %d:\t", __FILE__, __LINE__); printf("%s!\n", cudaGetErrorString(error)); \
        } \
  } while(0)
#else
#define CHECK_CUDA_ERROR()
#endif

// End debug-flags etc


// Note - just realized this is obligatory!
#define EVEN_ODD        1





#ifdef CUDA
#include <cuda_runtime_api.h>
#include "cuda_forall.h"
#include "cuda_reduce.h"
#include "cuda_rnd.h"
#endif


#if !USE_CUDA
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif
#endif


#define OPP_DIR(DIR)    ((DIR) + NDIM < (2*NDIM) ? ((DIR) + NDIM) : ((DIR) - NDIM))

// Variables - global
static __device__   int d_nSites;
static              int h_nSites;
static __device__   int d_padNSites; // Padded up to nearest 32 boundary
static              int h_padNSites;



typedef struct coordType_s
{
  int x[NDIM];
} coordType;

// Fields themselves
#if ISING_SITES
static uint32* s_spins;
#endif
#if ISING_GAUGE
// NOTE: FROM NOW ON ALWAYS PASS IN POINTERS
// Accessing arrays of pointers on device is the most silly thing ever in CUDA
// And it seriosly sucks. Another way would be to use some global context-pointer...
static  uint32* s_spinLinks[NDIM];
#endif

// Ok - we need 1 + NHITS/32 random variables per update
// (we can use every bit in first step)
static __device__ uint32*  rnd_data = NULL;
static uint32* h_rnd_data = NULL;
static uint32* p_rnd_data = NULL;
// Neighbor arrays
//static int* h_nbs[NDIM*2];
//static __device__ int* d_nbs[NDIM*2];
static __device__ int* d_nbs;
static int* h_nbs;


// Function declarations:
static void*    devAlloc        (size_t size);
static void     devFree         (void* ptr);

static void     initRnd         (int seed);
static void     generateRnds    (void);
static void     initNBArrays    (void);

static void     initLattice     (int seed);
static void     updateLattice   (radix beta);
static void     measure         (radix beta);

template <typename MEASFUNTYPE, typename INPUTTYPE>
static radix    measureLattice  (MEASFUNTYPE measFun, INPUTTYPE input);


static inline __host__ __device__
int     getFieldVal     (uint32* field, int index);
static inline __host__ __device__
void    setFieldVal     (uint32* field, int index, int val);
static inline __host__ __device__
uint32  getFieldPack    (uint32* field, int index);
static inline __host__ __device__
void    setFieldPack    (uint32* field, int index, uint32 val);

static inline
int     coordToIndex    (coordType coords);
static inline __host__ __device__
int     getNBId         (int i, int dir, int up);
static inline
int     getHNBId        (int i, int dir, int up);
static inline
void    setNBId         (int i, int dir, int up, int val);
static inline __host__ __device__
uint32  getRnd          (int nth, int metaSiteIdx);
static inline __host__ __device__
void    updateSitePackMP(int i, updateInput input);

#if DEBUG_NB_ARRAYS
static void checkNBArrays(void);
#endif


#if USE_CUDA
void* devAlloc(size_t size)
{
  void* result = NULL;
  cudaMalloc(&result,size);
  return result;

  }
void devFree(void* ptr)
{
  if (ptr) cudaFree(ptr);
}
#else
void* devAlloc(size_t size){
  void* result = NULL;
  result = malloc(size);
  return result;
}
void devFree(void* ptr)
{
  if (ptr) free(ptr);
}
#endif



int getFieldVal(uint32* field, int index)
{
  int fetchIdx = index >> 5; // index / 32
  int subIdx = index & 31;   // index % 32
  uint32 val = field[fetchIdx];
  // fetch bit
  return (val & (1 << subIdx)) != 0;
}

void setFieldVal(uint32* field, int index, int val)
{
  int fetchIdx = index >> 5; // index / 32
  int subIdx = index & 31;   // index % 32
  // Fetch and clear
  uint32 tmp = field[fetchIdx] & (~(1 << subIdx));
  // Set bit conditionally
  if (val) tmp |= (1 << subIdx);
  // Store
  field[fetchIdx] = tmp;
}

uint32 getFieldPack(uint32* field, int index){
  int fetchIdx = index >> 5; // index / 32
  return field[fetchIdx];
}

void setFieldPack(uint32* field, int index, uint32 val)
{
  int fetchIdx = index >> 5; // index / 32
  field[fetchIdx] = val;
}


int coordToIndex(coordType coords)
{
  int dir;
  int result = coords.x[0];
  int stride = DIMS[0];
#if EVEN_ODD
  int coordSum = coords.x[0];
#endif
  #ifdef CUDA
  #pragma unroll
  #endif
  for (dir = 1; dir < NDIM; dir++)
  {
    result += coords.x[dir] * stride;
    stride *= DIMS[dir];
#if EVEN_ODD
    coordSum += coords.x[dir];
#endif
  }
  // The site is an Even site if the sum of its coordinates is even
#if EVEN_ODD
  if ((coordSum & 0x1) == 0){
      result = result >> 1;
  } else {
      result = (result >> 1) + (h_nSites >> 1);
  }
#endif
  return result;
}


int getHNBId (int i, int dir, int up)
{
    if (up)
      dir = OPP_DIR(dir);
    return h_nbs[i * 2 * NDIM + dir];
}

int getNBId(int i, int dir, int up)
{
  if (up)
    dir = OPP_DIR(dir);
  return d_nbs[i * 2 * NDIM + dir];
}

void setNBId(int i, int dir, int up, int val)
{
  if (up)
    dir = OPP_DIR(dir);
  h_nbs[i * 2 * NDIM + dir] = val;
}




// How should we use random numbers?
// How about:
// Meta-site 'i' has random numbers { i, i + nMetaSites, i + 2*nMetaSites, ... } for itself?

uint32 getRnd(int nth, int metaSiteIdx)
{
    uint32 result;
#if USE_CUDA && USE_NEW_RND && (!PREGEN_NEW_RND)
    result = GENERATE_RND();
#elif USE_NEW_RND && (!USE_CUDA)
    result = mersenne_int(NULL);

#else
    result = rnd_data[nth * (d_padNSites >> 5) + metaSiteIdx];
#endif
    //uint32 x = nth * (d_padNSites >> 5) + metaSiteIdx;
    //return x * 0x9e3779b9u + 0xfa12345u;
/*    uint32 result = lrand48();
    uint32 result2 = lrand48();
    result = ((result & 0x00FFFF00u) >> 8) | ((result2 & 0x00FFFF00u) << 8);
    //if ((lrand48() & 0x1) != 0) result |= (1 << 31);
    return result;*/

    return result;
}




#if ISING_GAUGE

static inline __device__ __host__
int calcStapleSum(int site, int linkdir, uint32* spinLinks[NDIM])
{
    int staplesum = 0;
    int mu, nu;
    mu = linkdir;
    for (nu = 0; nu < NDIM; nu++){
        if (nu != mu){
            // The links are for upper staple: s_nu(x+mu), s_(-mu)(x+nu) and s_(-nu)(x)
            // And for lower staple: s_(-nu)(x+mu-nu), s_(-mu)(x-nu) and s_(nu)(x-nu)
            // Note that link inversion does nothing with Abelian gauges
            int l1, l2, l3, x;

            // First upper-staple
            x = getNBId(site, mu, 1);
            l1 = getFieldVal(spinLinks[nu], x);
            x = getNBId(site, nu, 1);
            l2 = getFieldVal(spinLinks[mu], x);
            l3 = getFieldVal(spinLinks[nu], site);
            // l1 + l2 + l3 tells number of positive links - if there are 1 or 3, then
            // l1*l2*l3 = 1 otherwise l1*l2*l3 = -1;
            staplesum += (((l1 + l2 + l3) & 0x1) == 0 ? -1 : 1);

            // Then the lower staple
            x = getNBId(site, nu, 0);          // Go to x - nu
            l1 = getFieldVal(spinLinks[nu], x); // Get s_nu(x-nu)
            l2 = getFieldVal(spinLinks[mu], x); // Get s_-mu(x-nu)
            x = getNBId(x, mu, 1);              // Go to x - nu + mu
            l3 = getFieldVal(spinLinks[nu], x); // Get s_-nu(x-nu+mu)
            staplesum += (((l1 + l2 + l3) & 0x1) == 0 ? -1 : 1);
        }
    }
    return staplesum;
}
#endif

static inline __device__ __host__
uint32 metropolisSumPack(uint32 newSpins, uint32 oldSpins, int i, int hit, radix beta, uint32* nbsums)
{
#if ISING_SITES
    beta *= 0.5;
#endif
    int counter;
    uint32 resultMask = 0;
    // Note - manual unroll of 4 - this should offer superb ILP (instruction level parallelism,
    // needed in pipelined (all) architectures). Also less occupancy needed to reach good performance.
    for (counter = 0; counter < 32; counter += 4){
      radix oldSpin0 = (oldSpins & (1 << (counter + 0))) != 0 ? 1.0f : -1.0f;
      radix newSpin0 = (newSpins & (1 << (counter + 0))) != 0 ? 1.0f : -1.0f;
      radix oldSpin1 = (oldSpins & (1 << (counter + 1))) != 0 ? 1.0f : -1.0f;
      radix newSpin1 = (newSpins & (1 << (counter + 1))) != 0 ? 1.0f : -1.0f;
      radix oldSpin2 = (oldSpins & (1 << (counter + 2))) != 0 ? 1.0f : -1.0f;
      radix newSpin2 = (newSpins & (1 << (counter + 2))) != 0 ? 1.0f : -1.0f;
      radix oldSpin3 = (oldSpins & (1 << (counter + 3))) != 0 ? 1.0f : -1.0f;
      radix newSpin3 = (newSpins & (1 << (counter + 3))) != 0 ? 1.0f : -1.0f;
      radix localDS[4];

      radix deltaSpin0 = newSpin0 - oldSpin0;
      radix deltaSpin1 = newSpin1 - oldSpin1;
      radix deltaSpin2 = newSpin2 - oldSpin2;
      radix deltaSpin3 = newSpin3 - oldSpin3;
      // Staplesums are packed in 4-bits, mask for lowest for bits is 15 = 0xF
      // Note also that the number was packed so that the representation was from 0 to 2x(NDIM-1)
      // Example in 3d, staplesum can be {-4, -2, 0, 2, 4 } and we pack it as { 0, 1, 2, 3, 4 }
      // Therefore to unpack, we need to subtract by (NDIM-1) and multiply by 2.
#if ISING_GAUGE
      #define COMP_TMP(X) \
            ((int)(((X) >> counter) & 0xF) << 1) - ((NDIM - 1) << 1)
#else
#define COMP_TMP(X) \
            ((int)(((X) >> counter) & 0xF) << 1) - ((NDIM) << 1)
#endif

      //int tmp1 = ((int)((nbsums[0] >> counter) & 0xF) << 1) - ((NDIM - 1) << 1);
      int tmp1 = COMP_TMP(nbsums[0]);
      radix stapleSum0 = (radix)tmp1;
      //int tmp2 = ((int)((nbsums[1] >> counter) & 0xF) << 1)  - ((NDIM - 1) << 1);
      int tmp2 = COMP_TMP(nbsums[1]);
      radix stapleSum1 = (radix)tmp2;
      //int tmp3 = ((int)((nbsums[2] >> counter) & 0xF) << 1)  - ((NDIM - 1) << 1);
      int tmp3 = COMP_TMP(nbsums[2]);
      radix stapleSum2 = (radix)tmp3;
      //int tmp4 = ((int)((nbsums[3] >> counter) & 0xF) << 1)  - ((NDIM - 1) << 1);
      int tmp4 = COMP_TMP(nbsums[3]);
      radix stapleSum3 = (radix)tmp4;
#undef COMP_TMP

      localDS[0] = -beta * deltaSpin0 * stapleSum0;
      localDS[1] = -beta * deltaSpin1 * stapleSum1;
      localDS[2] = -beta * deltaSpin2 * stapleSum2;
      localDS[3] = -beta * deltaSpin3 * stapleSum3;
      // Now we now deltaS's (local changes) - now we can do the Metropolis acceptance tests
      {
        int sub;
        #ifdef CUDA
        #pragma unroll
        #endif
        for (sub = 0; sub < 4; sub++)
        {
            int accept = 0;
            int locSite = counter + sub;
#if 1 // (USE_NEW_RND && USE_CUDA && !PREGEN_NEW_RND)
            uint32 rnd = getRnd(locSite + 1 + hit * 33, i);
#endif
            if (localDS[sub] <= 0.0f)
            {
                accept = 1;
            }
            else
            {
                // TODO: Better random numbers - and more cheap! :)
#if 0 // !(USE_NEW_RND && USE_CUDA && !PREGEN_NEW_RND)
                uint32 rnd = getRnd(locSite + 1 + hit * 33, i);
#endif
                // Note - we could avoid the "costly" division by doing multiplication on the other
                // side of the comparison
                //radix x = (radix)rnd / (radix)0xFFFFFFFFu;
                radix x = (radix)rnd / (radix)0xFFFFFFFFu;

                // Ok - accept if e^{-Delta S } > x ~ Uniform[0,1]
                if (expf(-localDS[sub]) > x)
                  accept = 1;
            }
            if (accept) resultMask |= (1 << locSite);
        }
      }
    }
    return resultMask;
}




#if ISING_SITES
static inline __device__ __host__
int calcSpinSum(int site, uint32* spins)
{
    int spinsum = 0;
    int nu;
    for (nu = 0; nu < NDIM; nu++){
        // This is simple: The action is Sum_<ij> s_i s_j = Sum_i s_i Sum_<j> s_j, so we
        // only need to compute the sum of the neighboring spins
        int x, s1, s2;
        // First upper-sping
        x = getNBId(site, nu, 1);
        s1 = getFieldVal(spins, x);
        x = getNBId(site, nu, 0);
        s2 = getFieldVal(spins, x);
        spinsum += (s1 + s2) * 2 - 2;
    }
    return spinsum;
}



#endif

void updateSitePackMP(int i, updateInput input)
{
  radix beta = input.beta;

  // How did metropolis work? Make a symmetric random change
  // and accept it if action is reduced always and if not, then
  // with probability e^{-DeltaS}, in our case S = -Beta * Sum_ij s_i * s_j

  // NOTE: We do not support coupling Ising-sitespins to Ising-gauge (at least, not yet)

  // First code for the site-spins

#if ISING_SITES

  // Ok - i is a 32-site pack index so deal with it

  // Sites packed in sets of 32 - one thread per 32 sites.
  int site = (i << 5);

  // Read the sites in:
  uint32 locSpins = getFieldPack(input.spins, site);


#if 0 //(NDIM > 8)
  uint32 spinsums[32] = { 0 };;
# error Broken codepath. Sorry, But for now I do not worry about more than 8 dimensions
#else
  uint32 spinsums[4] = { 0, 0, 0, 0 };
#endif

  int counter = 0;
  int locSite = site;
  // First compute the sums of the surrounding staples - these will be constant throughout the
  // local updates so we compute them once.
  //for (locSite = site; locSite < site + 32; locSite+= 4){
  for (counter = 0; counter < 32 ; counter += 4){
      int spinsum0 = calcSpinSum(locSite + 0,  input.spins);
      int spinsum1 = calcSpinSum(locSite + 1,  input.spins);
      int spinsum2 = calcSpinSum(locSite + 2,  input.spins);
      int spinsum3 = calcSpinSum(locSite + 3,  input.spins);
      int packed;
      packed = (spinsum0 >> 1) + NDIM;
      spinsums[0] |= packed << counter;
      packed = (spinsum1 >> 1) + NDIM;
      spinsums[1] |= packed << counter;
      packed = (spinsum2 >> 1) + NDIM;
      spinsums[2] |= packed << counter;
      packed = (spinsum3 >> 1) + NDIM;
      spinsums[3] |= packed << counter;
      locSite += 4;
  }
  // Then do the actual updates using the staplesums:
  // Do the hits
  {
      int hit;
      for (hit = 0; hit < NHITS; hit++)
      {
        // Ok we have rnd-numbers {i, i + d_padNSites/32, i + 2 * d_padNSites/32, ... }
        // So here we use one 32-bit rnd-number and in metropolisPack we use 32 (we handle 32 sites)
        // Let's just divide the rnd-numbers into two groups: [0, nMetaSites * nHits - 1] and
        // [nMetaSites * nHits, nMetaSites * 33 * nHits - 1]
        uint32 rnd = getRnd(33 * hit, i);
        uint32 newSpins;
        // Apply randomization of our spins - all 32 at once:
        newSpins = locSpins ^ rnd;
        // Compute deltaS's and do individual reject/accept tests
        uint32 acceptMask = metropolisSumPack(newSpins, locSpins, i, hit, beta, spinsums);
        locSpins = (locSpins & (~acceptMask)) | (newSpins & acceptMask);
      }
      setFieldPack(input.spins, site, locSpins);
  }

  /*
  // Do the hits
  int hit;
  for (hit = 0; hit < NHITS; hit++)
  {
    // Ok we have rnd-numbers {i, i + d_padNSites/32, i + 2 * d_padNSites/32, ... }
    // So here we use one 32-bit rnd-number and in metropolisPack we use 32 (we handle 32 sites)
    // Let's just divide the rnd-numbers into two groups: [0, nMetaSites * nHits - 1] and
    // [nMetaSites * nHits, nMetaSites * 33 * nHits - 1]
    uint32 rnd = getRnd(33 * hit, i);
    uint32 newSpins;
    // Apply randomization of our spins - all 32 at once:
    newSpins = locSpins ^ rnd;
    // Compute deltaS's and do individual reject/accept tests
    uint32 acceptMask = metropolisPack(newSpins, locSpins, i, hit, beta);
    locSpins = (locSpins & (~acceptMask)) | (newSpins & acceptMask);
  }
  */
  setFieldPack(input.spins, i << 5, locSpins);
#endif // ISING_SITES

#if ISING_GAUGE

  // Sites packed in sets of 32 - one thread per 32 sites.
  int dir = input.dir;
  int site = (i << 5);
  //int locSite;

  // Ok - let's settle for a compromise here - if there are 9 or more dimensions  then use 32-bits
  // per staplesum (with 9 dimensions you have 2 x 8 = 16 staples, which means you can have as staplesum
  // the number { -16, -14, -12, -10, -8 -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16 } =
  //            {   0,   1,   2,   3,  4, 5,  6,  7, 8, 9, 10,11,12,13, 14, 15, 16 }
  // So 2xNdim  - 1 options. So here you would need just the 5 bits to do it
  // Therefore at 8 or less dimensions use 4 bits to accumulate the staplesum into - this saves
  // registers and cache memory.
  // TODO: NDIM is best kept literal constant - as sizeof(x) cannot be evaluated at preprocessor
  // The problem is then of course that there should be a preprocessor operator which gives
  // length of an array - for now don't do more than 8 dimensions please...
#if 0 //(NDIM > 8)
  uint32 staplesums[32] = { 0 };;
# error Broken codepath. Sorry, But for now I do not worry about more than 8 dimensions
#else
  uint32 staplesums[4] = { 0, 0, 0, 0 };
#endif
  // Read the sites in:
  uint32 locSpins = getFieldPack(input.spinLinks[dir], site);
  int counter = 0;
  int locSite = site;
  // First compute the sums of the surrounding staples - these will be constant throughout the
  // local updates so we compute them once.
  //for (locSite = site; locSite < site + 32; locSite+= 4){
  for (counter = 0; counter < 32 ; counter += 4){
      int staplesum0 = calcStapleSum(locSite + 0, dir, input.spinLinks);
      int staplesum1 = calcStapleSum(locSite + 1, dir, input.spinLinks);
      int staplesum2 = calcStapleSum(locSite + 2, dir, input.spinLinks);
      int staplesum3 = calcStapleSum(locSite + 3, dir, input.spinLinks);
      int packed;
      packed = (staplesum0 >> 1) + NDIM - 1;
      staplesums[0] |= packed << counter;
      packed = (staplesum1 >> 1) + NDIM - 1;
      staplesums[1] |= packed << counter;
      packed = (staplesum2 >> 1) + NDIM - 1;
      staplesums[2] |= packed << counter;
      packed = (staplesum3 >> 1) + NDIM - 1;
      staplesums[3] |= packed << counter;
      locSite += 4;
  }
  // Then do the actual updates using the staplesums:
  // Do the hits
  {
      int hit;
      for (hit = 0; hit < NHITS; hit++)
      {
        // Ok we have rnd-numbers {i, i + d_padNSites/32, i + 2 * d_padNSites/32, ... }
        // So here we use one 32-bit rnd-number and in metropolisPack we use 32 (we handle 32 sites)
        // Let's just divide the rnd-numbers into two groups: [0, nMetaSites * nHits - 1] and
        // [nMetaSites * nHits, nMetaSites * 33 * nHits - 1]
        uint32 rnd = getRnd(33 * hit, i);
        uint32 newSpins;
        // Apply randomization of our spins - all 32 at once:
        newSpins = locSpins ^ rnd;
        // Compute deltaS's and do individual reject/accept tests
        uint32 acceptMask = metropolisSumPack(newSpins, locSpins, i, hit, beta, staplesums);
        locSpins = (locSpins & (~acceptMask)) | (newSpins & acceptMask);
      }
      setFieldPack(input.spinLinks[dir], site, locSpins);
  }
#endif
}





struct updateFields
{
  __host__ __device__
  void operator() (updateInput input, int metaIndex){
    updateSitePackMP(metaIndex, input);
  }
};

void initRnd(int seed)
{
  srand48(seed);
  seed_mersenne(seed);
}

#if USE_CUDA
#define COPY_VAR_TO_DEV(D_VAR, VAR)   cudaMemcpyToSymbol(D_VAR, &VAR, sizeof(VAR))
#else
#define COPY_VAR_TO_DEV(D_VAR, VAR)   do { D_VAR = VAR; } while (0)
#endif


void generateRnds(void)
{
    // We need in total NHITS x NMETASITES * 33 random numbers
    // How practical lrand48() gives uniform distribution in [0, 2^31 = (1 << 31)[ whereas we need
    // [0, 2^32 -1 = 0XFFFFFFFFu [ ... Well, let's just live without two bits and multiply lrand48() by two
    int i;
    if (!h_rnd_data){
        h_rnd_data = (uint32*)malloc(sizeof(uint32) * (h_padNSites >> 5) * NHITS * 33);
#if USE_CUDA
        p_rnd_data = (uint32*)devAlloc(sizeof(uint32) * (h_padNSites >> 5) * NHITS * 33);
        COPY_VAR_TO_DEV(rnd_data, p_rnd_data);
        //cudaMemcpy(&rnd_data, &d_rnd, sizeof(uint32*), cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR();
#else
        rnd_data = h_rnd_data;
#endif
    }
#if USE_CUDA && PREGEN_NEW_RND
    cudaGenerateRnds((h_padNSites >> 5) * NHITS * 33, p_rnd_data);
    CHECK_CUDA_ERROR();
    return;
#endif

#if !USE_NEW_RND
    for (i = 0; i < (h_padNSites >> 5) * NHITS * 33; i++){
        /*uint32 result = lrand48();
        uint32 result2 = lrand48();
        result = ((result & 0x00FFFF00u) >> 8) | ((result2 & 0x00FFFF00u) << 8);*/
        uint32 result = mersenne_int(NULL);
        h_rnd_data[i] = result;

    }
#endif
#if USE_CUDA
    cudaMemcpy(p_rnd_data, h_rnd_data, sizeof(uint32) * (h_padNSites >> 5) * NHITS * 33, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
#endif

}




void initNBArrays(void)
{
    int dir;
    int* tmpptr;
    h_nbs = (int*)malloc(sizeof(int) * h_padNSites * 2 * NDIM);
#if USE_CUDA
    tmpptr = (int*)devAlloc(sizeof(int) * h_padNSites * 2 * NDIM);
    COPY_VAR_TO_DEV(d_nbs, tmpptr);
#else
    COPY_VAR_TO_DEV(d_nbs, h_nbs);
#endif

    for (dir = 0; dir < NDIM; dir++){
        // Ok - arrays for the two directions have been allocated, now
        // fill them
        coordType x = {{0}};
        int done = 0;
        // Walk over lattice and fill in the neighbor indices
        while (!done){
            coordType y = x;
            int nbIndex;
            int ourIndex = coordToIndex(x);
            y.x[dir] += 1;
            if (y.x[dir] >= DIMS[dir]) y.x[dir] = 0;
            nbIndex = coordToIndex(y);
            setNBId(ourIndex, dir, 0, nbIndex);
            //h_nbs[dir][ourIndex] = nbIndex;

            y.x[dir] = x.x[dir] - 1;
            if (y.x[dir] < 0) y.x[dir] = DIMS[dir] - 1;
            nbIndex = coordToIndex(y);
            setNBId(ourIndex, dir, 1, nbIndex);
            //h_nbs[dir+ NDIM][ourIndex] = nbIndex;

            // Ok proceed to next coordinate value:
            {
                int mu;
                for (mu = 0; mu < NDIM; mu++){
                    x.x[mu]++;
                    if (x.x[mu] < DIMS[mu]) break;
                    // else
                    x.x[mu] = 0;
                    if (mu == NDIM - 1) done = true;
                }
            }
        }
    }
    // Ok now we have the arrays done on host side - copy it to device
#if USE_CUDA
    cudaMemcpy(tmpptr, h_nbs, sizeof(int) * h_padNSites * 2 * NDIM, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
#endif

}

// TODO: Fix this below
#if DEBUG_NB_ARRAYS

void checkNBArrays(void){
    // Ok - first check - each site should have two and exactly two neihgbours in each direction
    // find them
    {

        int i, j;
        for (i = 0; i < h_nSites; i++){
            int dir;
            int nNeigbours[2*NDIM] = { 0 };
            for (j = 0; j < h_nSites; j++){
                for (dir = 0; dir < 2*NDIM; dir++){
                    //if (h_nbs[dir][j] == i){
                    if (getHNBId(j, dir, 0) == i){
                        nNeigbours[dir]++;
                        // Also assert that we are its neighbor:
                        //assert(h_nbs[OPP_DIR(dir)][i] == j);
                        assert(getHNBId(i, OPP_DIR(dir), 0) == j);
                    }
                }
            }
            // Ok - check results:
            for (dir = 0; dir < NDIM * 2; dir++)
                assert(nNeigbours[dir] == 1);
            // Next check even-odd:
            #if EVEN_ODD
            for (dir = 0; dir < NDIM * 2; dir++){
                int weAreEven = i < (h_nSites >> 1);
                //int theyAreOdd = h_nbs[dir][i] >= (h_nSites >> 1);
                int theyAreOdd = getHNBId(i, dir, 0) >= (h_nSites >> 1);
                assert(weAreEven == theyAreOdd);
            }
            #endif

            // Then do a loop around a simple path and check that we end up were we came from
            for (dir = 0; dir < NDIM * 2; dir++){
                int dir2;
                for (dir2 = 0; dir2 < NDIM * 2; dir2++) if ((dir2 % NDIM) != (dir % NDIM)){
                  int tmpidx = i; // Start location
                  int oppdir = OPP_DIR(dir); // Note: oppdir definition not optimal
                  int oppdir2 = OPP_DIR(dir2);
                  tmpidx = h_nbs[dir][tmpidx]; // go forward                        o---o
                  tmpidx = h_nbs[dir][tmpidx]; // go forward                        |   |
                  tmpidx = h_nbs[dir2][tmpidx]; // turn "up"                o---x---o---o
                  tmpidx = h_nbs[oppdir][tmpidx]; // come back              |   .   |
                  tmpidx = h_nbs[oppdir2][tmpidx]; // Start crossing path   o---O---o
                  tmpidx = h_nbs[oppdir2][tmpidx]; // Keep crossing path
                  tmpidx = h_nbs[oppdir][tmpidx]; // come back
                  assert(h_nbs[dir2][tmpidx] == i); // Check that we can see starting point up 'O'
                  tmpidx = h_nbs[oppdir][tmpidx]; // come back
                  tmpidx = h_nbs[dir2][tmpidx]; // turn "up"
                  tmpidx = h_nbs[dir][tmpidx]; // go forward to starting position
                  // And assert that we came back to where we started
                  assert(i == tmpidx);
                }
                // Another check - let's walk through the boundaries and check that we come back
                {
                  int k;
                  int tmpidx = i;
                  for (k = 0; k < DIMS[dir % NDIM]; k++){
                    tmpidx = h_nbs[dir][tmpidx];
                  }
                  assert(i == tmpidx);
                }
                // Ok - I guess we have the nb-arrays ok
            }
        }
    }
}
#endif

void initLattice (int seed)
{
  int dir;
  uint32* tmpptr;
  initRnd(seed);
  h_nSites = DIMS[0];
  for (dir = 1; dir < NDIM; dir++)
    h_nSites *= DIMS[dir];
  // How to pad? Add one minus limit and mask out lowest bits
  h_padNSites = (h_nSites + 31) & (~31);

  COPY_VAR_TO_DEV(d_nSites, h_nSites);
  CHECK_CUDA_ERROR();
  COPY_VAR_TO_DEV(d_padNSites, h_padNSites);
  CHECK_CUDA_ERROR();

  // Field Allocations!

#if ISING_SITES
  // Allocate the field on device:
  tmpptr = (uint32*)devAlloc((h_padNSites >> 5) * sizeof(uint32));
  s_spins = tmpptr;
  CHECK_CUDA_ERROR();
#endif

#if ISING_GAUGE
  // Allocate the field on device:
  {
      int dir;
      for (dir = 0; dir < NDIM; dir++){
          unsigned int start = 0;
          tmpptr = (uint32*)devAlloc((h_padNSites >> 5) * sizeof(uint32) * NDIM);
          s_spinLinks[dir] = tmpptr;
          CHECK_CUDA_ERROR();
#if HOT_START
          start = 0xAAAAAAAAu;
#endif
#if USE_CUDA
          cudaMemset(tmpptr, start, (h_padNSites >> 5) * sizeof(uint32));
          CHECK_CUDA_ERROR();
#else
          memset(tmpptr, start, (h_padNSites >> 5) * sizeof(uint32));
#endif
      }
  }
#endif // ISING_GAUGE



  // Allocate and initialize neighbor arrays
  initNBArrays();
#if DEBUG_NB_ARRAYS
  // Sanity check them
  checkNBArrays();
#endif

#if ISING_SITES
  unsigned int start = 0;
#if HOT_START
  start = 0xAAAAAAAAu;
#endif
  // For now init lattice to cold lattice (ie. clear all bits)
#if USE_CUDA
  cudaMemset(tmpptr, start, (h_padNSites >> 5) * sizeof(uint32));
#else
  memset(s_spins, start, (h_padNSites >> 5) * sizeof(uint32));
#endif
#endif
  CHECK_CUDA_ERROR();
}


struct radixSum{
    __host__ __device__
    radix operator ()(radix a, radix b){
        return a+b;
    }
};

template <typename MEASFUNTYPE, typename INPUTTYPE>
radix measureLattice(MEASFUNTYPE measFun, INPUTTYPE input)
{
    radix result = 0.0f;
    struct radixSum sumFun;
#if USE_CUDA
    callReduceKernel(input, measFun, sumFun, 0, h_nSites, &result);
#else
    int i;
    for (i = 0; i < h_nSites; i++){
        result = sumFun(result, measFun(input, i));
    }
#endif
    CHECK_CUDA_ERROR();
    return result;
}

void updateLattice(radix beta)
{
  struct updateFields updateFun;
  int metaSite;
  updateInput input;
  input.beta = beta;
#if ISING_GAUGE
  {
    int dir;
    for (dir = 0; dir < NDIM; dir++) input.spinLinks[dir] = s_spinLinks[dir];
    for (dir = 0; dir < NDIM; dir++){
        input.dir = dir;

#endif

#if ISING_SITES
    input.spins = s_spins;
#endif

#if USE_CUDA
    (void)metaSite;
    #if (USE_NEW_RND == 0) || (PREGEN_NEW_RND)
    generateRnds();
    CHECK_CUDA_ERROR();
    #endif

    // Do first EVEN then ODD sites
#if PREGEN_NEW_RND
    callTransformKernel(input, updateFun, 0, (h_padNSites >> 5) >> 1);
    callTransformKernel(input, updateFun, (h_padNSites >> 5) >> 1, h_padNSites >> 5);
#else
    callxFormKernel_UseRnds(input, updateFun, 0, (h_padNSites >> 5) >> 1);
    callxFormKernel_UseRnds(input, updateFun, (h_padNSites >> 5) >> 1, h_padNSites >> 5);
#endif
#else // if USE_CUDA
    // Create random variables
    generateRnds();
    for (metaSite = 0; metaSite < (h_padNSites >> 5); metaSite++)
      updateFun(input, metaSite);
#endif // if USE_CUDA
#if ISING_GAUGE
    }
  }
#endif
  CHECK_CUDA_ERROR();
}


#if ISING_GAUGE

typedef struct measInput_s
{
    uint32* spinLinks[NDIM];
} measInput;

struct measPlaquetteFun{
    __host__ __device__
    radix operator() (measInput input, int i, int skip = 0){
        // Note: Now i is a real site-index
        int mu, nu;
        radix plaquette = 0;
        for (nu = 1; nu < NDIM; nu++)
        for (mu = 0; mu < nu; mu++){
            int link = getFieldVal(input.spinLinks[mu], i) != 0 ? 1 : 0;
            int l1, l2, l3, x;


            // First upper-staple
            x = getNBId(i, mu, 1);
            l1 = getFieldVal(input.spinLinks[nu], x);
            x = getNBId(i, nu, 1);
            l2 = getFieldVal(input.spinLinks[mu], x);
            l3 = getFieldVal(input.spinLinks[nu], i);

            // Even number of positive links implies that the product is positive
            plaquette += ((l1+l2+l3+link) & 0x1) == 0 ? 1.0f : -1.0f;
        }
        return plaquette;
    }
};


void measure(radix beta)
{
    static int nthMeas = 0;

    double invVol = 1.0 / (double)h_nSites;
    double invPlaqs = invVol / (double)(NDIM);
    double plaquette;

    if (nthMeas == 0){
        printf("s(i.j;x) = s_i(x) s_j(x+i) s_-i(x+j) s_-j(x)\t");
    }

    {
        struct measPlaquetteFun plaquetteFun;
        measInput input;
        int dir;
        for (dir = 0; dir < NDIM; dir++) input.spinLinks[dir] = s_spinLinks[dir];
        plaquette = invPlaqs * (double)measureLattice(plaquetteFun, input);
        printf("%f\t", plaquette);
    }
    printf("\n");
    nthMeas++;
}
#endif


#if ISING_SITES
struct measEnergyFun{
    __host__ __device__
    radix operator() (unsigned int* spins, int i, int skip = 0){
        // Note: Now i is a real site-index
        radix ourSpin = getFieldVal(spins, i) != 0 ? 1.0f : -1.0f;
        int dir;
        radix result = 0.0f;
        for (dir = 0; dir < NDIM; dir++){
            int nbIdx = getNBId(i, dir, 1);
            radix nbSpin = getFieldVal(spins, nbIdx) != 0 ? 1.0f : -1.0f;
            result += ourSpin * nbSpin;
        }
        return result;
    }
};


struct measMagnFun{
    __host__ __device__
    radix operator() (unsigned int* spins, int i, int skip = 0){
        // Note: Now i is a real site-index
        radix ourSpin = getFieldVal(spins, i) != 0 ? 1.0f : -1.0f;
        return ourSpin;
    }
};

static double* s_energies = NULL;
static double* s_magnets = NULL;

void measure(radix beta)
{
    static int nthMeas = 0;

    double invVol = 1.0 / (double)h_nSites;
    double energy;
    double magnetization;

    if (nthMeas == 0){
        //printf("S=-beta/2 * Energy\t");
        printf("s_i s_(i+mu) / Link\t");
        printf("Magnetization\t");
        printf("|M|\t");
        printf("M^2\t");
        printf("S/|M|\n");

    }

    {

        struct measEnergyFun energyFun;
        //energy = -beta * 0.5 * (double)measureLattice(energyFun, s_spins);
        energy =  (double)(2.0 -  measureLattice(energyFun, s_spins)  * invVol);
        s_energies[nthMeas] = energy;
        printf("%f\t", energy);
    }

    {

        struct measMagnFun magnFun;
        magnetization = invVol * (double)measureLattice(magnFun, s_spins);
        s_magnets[nthMeas] = magnetization;
        printf("%f\t", magnetization);
        printf("%f\t", fabs(magnetization));
        printf("%f\t", magnetization * magnetization);
    }
    printf("%f\t", energy/fabs(magnetization));


    printf("\n");
    nthMeas++;
}
#endif

template <typename TYPE>
struct normalInputFun{
  TYPE operator()(TYPE* x, int i) const{
    return x[i];
  }
};

template <typename TYPE>
struct absInputFun{
  TYPE operator()(TYPE* x, int i){
    return fabs(x[i]);
  }
};

// NOTE: We put the volume here!
template <typename TYPE>
struct devianceInputFun{
  TYPE avg;
  TYPE operator()(TYPE* x, int i){
    double Vol = (double)h_nSites;
    TYPE tmp = x[i] - avg;
    return Vol * tmp * tmp;
  }
};


struct normalInputFun<double> doubleGetFun;
struct devianceInputFun<double> devianceGetFun;
struct absInputFun<double> absGetFun;


// This is a more reliable way to sum lots of values together - be careful though, a lot of compilers
// fail to produce "correct" code out of this and they "optimize" the correction term away.
// With infinite precision, the correction term would always be zero.
template <bool printEntries, typename XFORMFUNTYPE, typename INDEXTYPE, typename RADIXTYPE, typename INPUTTYPE>
static RADIXTYPE kahan_sum(INPUTTYPE input, INDEXTYPE start, INDEXTYPE end, XFORMFUNTYPE getInput, const char* name = NULL)
{
  RADIXTYPE sum = (RADIXTYPE)0;
  RADIXTYPE c = (RADIXTYPE)0;
  INDEXTYPE i;
  if (name) printf("%s\n", name);
  for (i = start; i < end; i++){
    RADIXTYPE tmp = getInput(input, i);
    if (printEntries){
      printf("%f\n", tmp);
    }
    RADIXTYPE y = tmp - c;
    RADIXTYPE t = y + sum;
    c = (t - sum);
    c = c - y;
    sum = t;
  }
  return sum;
}


// Entrypoint
int main (int argc, char** argv)
{
    int seed = 1;
    int nTrajs = DEFAULT_TRAJS;
    radix beta = s_beta;
#ifdef CUDA
    printf("Usage: ./ising_gpu <float beta> <int nTrajectories> <int seed>\n\n");
#else
    printf("Usage: ./ising_cpu <float beta> <int nTrajectories> <int seed>\n\n");
#endif
    if (argc >= 2) beta = (radix)atof(argv[1]);
    if (argc >= 3) nTrajs = atoi(argv[2]);
    if (argc >= 4) seed = atoi(argv[3]);

    printf("Lattice size = [ %d", DIMS[0]);
    {
        int dir;
        for (dir = 1; dir < NDIM; dir++){
            printf(", %d", DIMS[dir]);
        }
    }
    printf(" ]\n");

    printf("Beta = %f\n", beta);
    printf("Num Trajectories = %d \t Hits per trajectory = %d\n", nTrajs, NHITS);
    printf("Seed = %d\n\n", seed);
#if USE_CUDA
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
#endif

    initLattice(seed);


#if ISING_SITES
    // Allocate measurement arrays:
    s_energies = (double*)malloc(nTrajs * sizeof(double));
    s_magnets =  (double*)malloc(nTrajs * sizeof(double));
#endif
    {
        // Actual runs
        int traj;
        // But first thermalize the lattice:
        printf("Thermalizing lattice:\n");
        for (traj = 0; traj < NTHERMAL; traj++){
            updateLattice(beta);
            if ((traj & 7) == 7){ printf("."); fflush(stdout);}
        }
        printf("Done!\nStarting updates with measure-runs:\n");
        // Then do runs with measurements
        for (traj = 0; traj < nTrajs; traj++){
            updateLattice(beta);
            measure(beta);
        }
    }
#if ISING_SITES
    // Compute averages
    {
        // ??? Compiler cannot deduce types automatically ??? Odd
        double energy = kahan_sum<false, normalInputFun<double>, int, double >
          (s_energies, 0, nTrajs, doubleGetFun);
        double magnetization = kahan_sum<false, normalInputFun<double>, int, double >
          (s_magnets, 0, nTrajs, doubleGetFun);
        devianceGetFun.avg = magnetization / (double)nTrajs;
        double susc = kahan_sum<false, devianceInputFun<double>, int, double >
          (s_magnets, 0, nTrajs, devianceGetFun, "Susceptibility");
        double absM = kahan_sum<false, absInputFun<double>, int, double>
          (s_magnets, 0, nTrajs, absGetFun);

        printf("Avg. Energy = %f\n", energy / (double)nTrajs);
        printf("Avg. Magnetization = %f\n", magnetization / (double)nTrajs);
        printf("Avg. |M| = %f\n", absM / (double)nTrajs);
        printf("Avg. Susceptibility = %f\n", susc / (double)nTrajs);
    }
#endif
    // Let os clean...
    return 0;
}
