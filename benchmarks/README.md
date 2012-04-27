
CUDA-Histogram Performance Benchmark Results
============================================

This folder/directory contains performance benchmark results for the CUDA-histogram project. The results are stored as plain text of comma-separated values in .csv files and are parsed and plotted using _gnuplot_ scripts.

Current Results
===============

 1) Benchmark results for test_perf.cu (see parent folder)
 - Runs normal histogram benchmarks of various bin-sizes (from 1 to 4096)
 - Three types of key-distributions:
  * Uniform random
  * Real texture data
  * Almost degenerate (repeat same index 100 times in a row)
 - Results stored (for now) in:
   * "perfred_degen.csv"
   * "perfred_load.csv"
   * "perfred_rnd.csv"
 - Plot your results using "gnuplot plot_perf.p"
   * Edit "plot_perf.p" to change terminal (defaults to png)
