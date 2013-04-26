 Generalized CUDA-histogram v0.1 README
----------------------------------------

Copyright Teemu Rantalaiho 2011-2012 - Email: "firstname"."lastname"@helsinki.fi

This file contains a quick introduction, as well as quick instructions on how to
use our generalized histogram code for CUDA-capable GPUs. The code is fast and
generic, and the abstraction has been achieved by using function objects as
template parameters. Therefore the API is C++, but with minimal C++ features.
What we mean by fast here, is that it wins known histogram algorithms for CUDA
by as much as 40% to 135% in common use-cases (for example the NVIDIA
Performance Primitives' 1 and 4-channel 256-bin histogram - tested on v.4.0.17).
And what we mean by generic is that it supports arbitrary bin-sizes and types,
which means that you can basically do any histogram-type of operation that comes
into your mind with this code. As long as you need to "sum up" some kind of
values into bins, this code can do it; You can also replace the sum by any
associative and commutative binary operation (we require the operator to commute
in order to compute the sums in fastest possible order)

What is it?
============

A generalized histogram can be achieved by the following reasoning:

A normal histogram is constructed by taking a set (array/vector) of
(integer-valued) indices that belong into some range [0, Nbins -1], and
computing how many times each index occurs in the set of input-indices. Then the
result of the histogram is an array of integers of the length Nbins, where
each entry tells how many times that particular index-value was present in the
input-set. Simply put the idea is to take always number one, and add it to a
bin that the user specifies. 

Suppose then that instead of adding the constant number one to each bin, you
would like to add up other values, possibly of other type than the integer
number (floating point numbers, vectors, whatever). This construction we call a
'Generalized histogram'. Essentially in our code the user passes in an
arbitrary input-context, a function-object, which takes in the input context and
input index and outputs a bin-index and output-value (actually there can
be multiple for just one input-entry) and another function-objects that defines
how two of those values can be summed up. After this our code takes care of
the rest.

Features
========

 - Fast normal histograms, up to 2560 bins (Optimized for Fermi, runs with all
    CUDA-cards)
 - Support for practically all histogram sizes: Tested up to 131072 bins
 - Support for arbitrary bin-types:
    * Want to add together complex numbers, or find the pixels with most red
        color in each bin? No problem: you write the function-object that sums
        up the entries
 - Multiple outputs per one input (faster execution on fermi, where 128-bit
    loads rule)
 - Enables advanced arithmetic for each input-entry - again, by letting the
    user define what are the ouput-indices and values for each input-index.
 - Output to host or device memory (For maximal flexibility)
 - Execute on user-given CUDA-stream (take advantage of concurrency)
 - Accumulates on top of previous arrays of results
    (Easy to accumulate multiple passes)
 - Support for user-given temporary-arrays (reduce latency - important in small
    problems, as GPU-memory allocation is a relatively heavy operation)

System requirements
===================

 - CUDA toolkit 3.2.16 or newer tested (may work with older also)
 - CUDA-capable GPU
 - Uses CUDA-runtime library

Installation
============

 - Code provided as one header file: cuda_histogram.h - just include it and go
    (To be compiled with nvcc)

Short Usage instructions
========================

 - Two public APIs with minimal template parameters:
   * Checking the size for the temporary buffer to use -- just pass in
        histogram type (generic versus normal), output-type and number of bins
        and this API tells how many bytes of temporary storage the actual
        histogram-call will need. Then if you want, you can pass your own
        temporary buffer to the actual histogram-call to reduce latency
   * Actual histogram call -- idea: You pass in an arbitrary input-context,
        input-index range and two function object that give the desired
        histogram values and keys (bin-indices) for each input-index and sum up
        output-values, and the histogram code does the rest.
 - See cuda_histogram.h header "callHistogramKernel()" and
        "getHistogramBufSize()" documentation for details
 - Also check out samples: "test_*.cu", especially test_minimal.cu
 - Performance benchmark results are stored in "benchmarks"-directory

Minimal test example
====================

    #include "cuda_histogram.h"
    #include <assert.h>
    #include <stdio.h>

    // minimal test - 1 key per input index
    struct test_xform {
      __host__ __device__
      void operator() (int* input, int i, int* res_idx, int* res, int nres) const {
        *res_idx++ = input[i];
        *res++ = 1;
      }
    };

    // Sum-functor to be used for reduction - just a normal sum of two integers
    struct test_sumfun {
        __device__ __host__ int operator() (int res1, int res2) const{
            return res1 + res2;
        }
    };

    // Entrypoint
    int main (int argc, char** argv)
    {
        // Allocate an array on CPU-side to fill data of 10 indices
        int h_data[] = { 0, 1, 0, 2, 2, 3, 1, 5, 0, 0 };
        // Allocate an array on GPU-memory to hold the input
        int* d_data = NULL;
        cudaMalloc(&d_data, 10 * sizeof(int));
        assert(d_data); // Probably something wrong with GPU initialization
                        // - check your CUDA runtime and drivers.
        // Copy the input-data to the GPU
        cudaMemcpy(d_data, h_data, 10 * sizeof(int), cudaMemcpyHostToDevice);

        // Init the result-array on host-side to zero (we have 6 bins, from 0 to 5):
        int res[6] = { 0 };

        // Create the necessary function objects and run histogram using them 
        // 1 result per input index, 6 bins, 10 inputs
        test_xform xform;
        test_sumfun sum;
        callHistogramKernel<histogram_atomic_inc, 1>
          (d_data, xform, sum, 0, 10, 0, &res[0], 6);

        // Print the result, should be: [ 4, 2, 2, 1, 0, 1 ]
        printf("Results:  [ %d, %d, %d, %d, %d, %d ]\n", 
                res[0], res[1], res[2], res[3], res[4], res[5]);
        printf("Expected: [ 4, 2, 2, 1, 0, 1 ]\n");
        // Done: Let OS + GPU-driver+runtime worry about resource-cleanup
        return 0;
    }

A Very Short Description of the Algorithms Used
===============================================

 - Histograms with a small number of bins:
    * Have a private histogram for each thread and accumulate
    * Gets you to about 64 bins of 32-bit data and to 256 bins of 8-bit
      normal histograms -- implementation follows 
      [Cedric Nugteren, Gert-Jan van den Braak, Henk Corporaal, and Bart Mesman. 2011. 
       High performance predictable histogramming on GPUs: 
           exploring and evaluating algorithm trade-offs. 
       In Proceedings of the Fourth Workshop on General Purpose Processing on Graphics 
       Processing Units (GPGPU-4). ACM, New York, NY, USA, , 
      Article 1 , 8 pages. DOI=10.1145/1964179.1964181 http://doi.acm.org/10.1145/1964179.1964181]
      Except that our implementation uses 8-bit thread-private histograms and has to accumulate
      to a threadblock-wide histogram every 255 steps.
    * Complexity is `O(n_in / n_p)`, where `n_in` is the number of inputs and `n_p` is the number of processors
    * With larger amount of bins less concurrent warps can be scheduled leading to under-utilization
      or processors, slightly hurting performance. The reason is that more shared memory is needed to satisfy
      more concurrent warps.
 - Histograms with a moderate amount of bins:
    * One histogram per block in shared memory.
    * Use an adaptive algorithm to help with collisions:
    (Once high amount of degeneracy found (~16 degenerate keys between 32 threads))
     * Do warp-local reduction of values in shared memory
     * One thread (per unique key) collects the sum
     * Write result, free of collisions
     * Expensive for well distributed keys → only apply when high degeneracy detected
    * Collisions with shared memory atomics cause serialization, which should result in `O(n_in * avg_coll / n_p)`
      complexity, but this is alleviated by using the adaptive algorithm -- here `avg_coll` is the 
      average amount of collisions per warp.
    * The complexity of the conflict resolution part is `O(n_unique)`, where `n_unique` is the number of 
      unique keys that the warp in question has, within its 32 keys (one key per thread) -- therefore
      the collision resolution-algorithm (as is) is only helpful when there is little variation in the data.
 - Large histograms:
    * At around 2500 bins we run low on shared memory
    * First solution: Multiple passes of the medium histogram algorithm
        * Improved occupancy and cache help
    * With very large histograms (~100 000 bins) too many passes:
        * Resort to global memory atomics
        * For generalized histograms use a per-warp hashtable in shared memory
        * Adaptive warp-reduction for degenerate key-distributions
        * Performance drops to about same level as thrust at around 100 000 bins
        * CPU should be roughly twice as fast here
        * Even as is, could be useful to use GPU:
            * Complex key-value resolving code can amortize (relatively) slow histogram code
        * Global atomics will be faster in Kepler:
            * Coupled with warp-reduction, could be very competitive



Performance
===========

 - For synthetic benchmark results consult the (wiki-page)[https://github.com/trantalaiho/Cuda-Histogram/wiki].
 - Other results with comparison to previous methods include (all results run on CUDA 4.0.17, Tesla M2070 ECC On):
 - Real Image histogram 8-bit (256-bin) grayscale histogram:
    * Source-code available (here)[https://github.com/trantalaiho/Cuda-Histogram/blob/master/tests_histogram/test_image_8b_1C.cu]
    * NVIDIA Performance Primitives: 8GK/s
    * Our implementation: 18GK/s (125% Improvement)
    * Different image sizes:
        * 1024x1024 -- NPP: 3.56GK/s, Our: 5.7GK/s (Improvement: 60%)
        * 2048x2048 -- NPP: 6.64GK/s, Our: 11.76GK/s (Improvement: 77%)
        * 4096x4096 -- NPP: 7.6GK/s, Our: 16.3GK/s (Improvement: 114%)
        * 8192x8192 -- NPP: 7.99GK/s, Our: 17.97GK/s (Improvement: 125%)
 -  Three-channel image histogram 8-bits per channel (256-bins per channel) RGB(A) histogram:
    * See: [https://github.com/trantalaiho/Cuda-Histogram/blob/master/tests_histogram/test_image_8b_4C.cu]
    * NVIDIA Performance Primitives: 6GK/s
    * Our implementation: 12GK/s (2x faster than NPP)
 - Generalized Histogram Performance:
    * See: [https://github.com/trantalaiho/Cuda-Histogram/blob/master/tests_histogram/test_sum_rows.cu]
    * Test-case: Sum over every row in a fp32 matrix
    * Thrust reduce-by-key ~ 1.8GK/s (normal reduction ~20GK/s = 80GB/s)
    * Our algorithm 2-7 times faster than thrust, depending on form of matrix (number of bins)
        * Matrix: 50 x 1 000 000 -- Sum Rows time: 4.73ms -> 10.58 GK/s
        * Matrix: 500 x 100 000 -- Sum Rows time: 4.06ms -> 12.33 GK/s
        * Matrix: 5000 x 10 000 -- Sum Rows time: 12.13ms -> 4.12 GK/s
 

     

Known Issues
============

 - Medium-sized histograms (~500 bins and more) use quite a lot of
    GPU-memory -- more than necessary -- This has been partially fixed now.
 - No byte-based bin fastpaths available for normal histograms of medium size
 - Large-sized histograms (~2000 bins and more) should be further optimized
    (Performance has doubled from initial commit, but is still not very good)
 - Compilation time is long (Too many kernels) -- Fixed
 - Code could use some cleaning up (it is still quite fresh out of the oven)
 - Optimization has been concentrated on Fermi-level hardware. Older hardware
     is supported, but little work has been done to optimize for it.
 - No optimizations for Kepler level hardware yet -- this is pending acquisition of
    said hardware.
 - Fastest way to do medium-sized histograms (For example 4x256 bins) is to
     do multiple passes over the data with smaller number of bins (256).
     See for example 'test_image_8b_4AC.cu'. This is due to the fact that
     each channel can be viewed as a sub-problem and in fact this problem
     is not a 1024-bin histogram problem, but 4 256-bin histogram problems, since
     the bin-range of each key is known beforehand to lie within one of the channels.
 - TODO: Implement register-based threadblock-wide accumulation histograms for small normal
   histograms, introduced in [Brown, S.; Snoeyink, J., 
      "Modestly faster histogram computations on GPUs," 
      Innovative Parallel Computing (InPar), 2012 , vol., no., pp.1,7, 13-14 May 2012]
      [URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6339589&isnumber=6339585]. 
   The implementation at the moment is very similar to theirs, except that is uses local
   memory for accumulation histograms, which is slower than registers, even in L1-cache.

