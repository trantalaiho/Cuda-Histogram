/*
 *
 *
 *  Created on: 5.4.2012
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

 // Ok This test is NOT self-contained - just output a raw file of random numbers and run it through dieharder
 // We should consider also trying TestU1

 // Compile with:  
 //   nvcc -O4 -arch=<your_arch> -I../ test_random.cu -o test_random
 

 #include "cuda_rnd.h"
 #include <stdlib.h>

#define THREADS_PER_BLOCK_LOG2      7
#define BLOCKS_PER_GRID_LOG2        6 // 2^13 = 8192

#define THREADS_PER_BLOCK           (1 << THREADS_PER_BLOCK_LOG2)
#define BLOCKS_PER_GRID             (1 << BLOCKS_PER_GRID_LOG2)
#define NSTREAMS                    (1 << (THREADS_PER_BLOCK_LOG2 + BLOCKS_PER_GRID_LOG2))


void printHelp(void)
 {
   printf("\n");
   printf("Test GPU random number generation - outputs a file called 'rnd.dat' by default \n\n");
   printf("\tOptions:\n\n");
   printf("\t\t--out <filename> Write to <filename> instead of 'rnd.dat'.\n");
   printf("\t\t--stream <id>\t Read a single parallel random stream <id> in [0, %d]\n", NSTREAMS - 1);
   printf("\t\t--safemode \t Generate random numbers in safemode (Allows divergent threads -- NOTE: NOT FUNCTIONAL YET! AVOID...)\n");
   printf("\t\t--nout <n> \t Generate <n> uint32 numbers 10 000 000 by default.\n");
   printf("\t\t--seed <seed> \t Provide a custom seed - otherwise use /dev/urandom/.\n");
 }

 int main (int argc, char** argv)
 {
   FILE* output;
   bool safemode = false;
   int streamID = -1;
   size_t nOut = 10 * 1000 * 1000;
   const char* out = "rnd.dat";
   int i;
   bool useSeed = false;
   int seed = 0;

   printHelp();
   for (i = 0; i < argc; i++)
   {
     if (argv[i] && strcmp(argv[i], "--safemode") == 0)
       safemode = true;

     if (argv[i] && strcmp(argv[i], "--out") == 0){
       i++;
       if (i >= argc) { printf("--out <filename>, parameter <filename> missing"); return -1; }
       out = argv[i];
     }

     if (argv[i] && strcmp(argv[i], "--stream") == 0){
       i++;
       if (i >= argc) { printf("--stream <id>, parameter <id> missing"); return -2; }
       streamID = atoi(argv[i]);
       if (streamID < 0 || streamID >= NSTREAMS){
         printf("Invalid streamID: %d must be within [0, %d] !!! \n", streamID, NSTREAMS - 1);
         return -3;
       }
     }

     if (argv[i] && strcmp(argv[i], "--nout") == 0){
       i++;
       if (i >= argc) { printf("--nout <n>, parameter <n> missing"); return -4; }
       nOut = atol(argv[i]);
     }

     if (argv[i] && strcmp(argv[i], "--seed") == 0){
       i++;
       if (i >= argc) { printf("--seed <seed>, parameter <seed> missing"); return -6; }
       seed = strtoul(argv[i], NULL, 0);
       useSeed = true;
     }
   }

   // Open output
   output = fopen(out, "wb+");
   if (!output){
     printf("Could not open file: %s for writing!\n", out);
     return -10;
   }

   if (streamID >= 0 || safemode)
   {
     // Here we use custom algorithm
   }
   else
   {
     // Here no special things - use default algo
     const size_t genLimit = 1024 * 1024 * 128; // Generate at most 128 MegaBinaryBytes at a time
     size_t gen = nOut > genLimit ? genLimit : nOut;
     size_t ngen = 0;
     unsigned int* result = NULL;
     unsigned int* h_result = (unsigned int*)malloc(sizeof(unsigned int) * gen);
     while (ngen < nOut){
       result = cudaGenerateRnds(gen, result, result == NULL, !useSeed, seed);
       if (!result){
         printf("cudaGenerateRnds() returned NULL - cudaError = %s\n", cudaGetErrorString(cudaGetLastError()));
       } else {

         cudaMemcpy(h_result, result, sizeof(unsigned int) * gen, cudaMemcpyDeviceToHost);
         if (fwrite(h_result, sizeof(unsigned int), gen, output) != gen)
           printf("Writing to file failed!");
       }
       ngen += gen;
       gen = nOut - ngen;
       if (gen > genLimit) gen = genLimit;
     }
     cudaFree(result);
     free(h_result);
   }
   fclose(output);
   return 0;
 }


