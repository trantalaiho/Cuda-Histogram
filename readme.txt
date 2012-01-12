 Generalized CUDA-histogram v0.1 README
----------------------------------------

Copyright Teemu Rantalaiho 2011

This file contains a quick introduction, as well as quick instructions on how
to use our generalized histogram code for CUDA-capable GPUs. The code is fast
and generic, and the abstraction has been achieved by using function objects
as template parameters. Therefore the API is C++, but with minimal C++
features.

What we mean by fast here, is that it wins known histogram algorithms for CUDA
by as much as 20% to 50% in common use-cases (for example the NVIDIA
Performance Primitives' 1-channel 256-bin histogram - tested on v.4.0.17).
And what we mean by generic is that it supports arbitrary bin-sizes and types,
which means that you can basically do any histogram-type of operation that
comes into your mind with this code. As long as you need to "sum up" some kind
of values into bins, this code can do it; You can also replace the sum by any
associative and commutative binary operation (we require the operator to
commute in order to compute the sums in fastest possible order)

What is it?
============

A generalized histogram can be achieved by the following reasoning:

A normal histogram is constructed by taking a set (array/vector) of
(integer-valued) indices that belong into some range [0, Nbins -1], and
computing how many times each index occurs in the set of input-indices. Then
the result of the histogram is an array of integers of the length Nbins, where
each entry tells how many times that particular index-value was present in the
input-set. Simply put the idea is to take always number one, and add it to a
bin that the user specifies.

Suppose then that instead of adding the constant number one to each bin, you
would like to add up other values, possibly of other type than the integer
number (floating point numbers, vectors, whatever). This construction we call
a 'Generalized histogram'. Essentially in our code the user passes in an
arbitrary input-context, a function-object, which takes in the input context
and input index and outputs a bin-index and output-value (actually there can
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
 - Also check out samples: "test_*.cu", especiall test_minimal.cu


Known Issues
============

 - This is the first release of this software, although we have run many tests
    there are bound to be many issues on it that we are not yet aware of -
    please file bugs on issues encountered
 - CudaStream-API largely untested
 - Medium-sized histograms (~500 bins and more) use quite a lot of
    GPU-memory -- more than necessary
 - No byte-based bin fastpaths available for normal histograms of medium size
    (~500 bins and more)
 - Large-sized histograms (~2000 bins and more) should be further optimized
 - Compilation time is long (Too many kernels)
 - Code could use some cleaning up (it is still quite fresh out of the oven)

