 Generalized CUDA-histogram v0.1 README
----------------------------------------

Copyright Teemu Rantalaiho 2011 - 2012
Email: "firstname"."lastname"@helsinki.fi

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

 - Fast normal histograms, up to 1024 bins (Optimized for Fermi, runs with all
    CUDA-cards)
 - Support for practically all histogram sizes: Tested up to 16384 bins
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


Known Issues
============

 - This is the first release of this software, although we have run many tests
    there are bound to be many issues on it that we are not yet aware of -
    please file bugs on issues encountered
 - CudaStream-API largely untested
 - Medium-sized histograms (~500 bins and more) use quite a lot of
    GPU-memory -- more than necessary -- This has been partially fixed now.
 - No byte-based bin fastpaths available for normal histograms of medium size
 - Large-sized histograms (~2000 bins and more) should be further optimized
    (Performance has doubled from initial commit, but is still not very good)
 - Compilation time is long (Too many kernels) -- Fixed
 - Code could use some cleaning up (it is still quite fresh out of the oven)
 - Optimization has been concentrated on Fermi-level hardware. Older hardware
     is supported, but little work has been done to optimize for it.
 - Fastest way to do medium-sized histograms (For example 4x256 bins) is to
     do multiple passes over the data with smaller number of bins (256).
     See for example 'test_image_8b_4AC.cu'.
