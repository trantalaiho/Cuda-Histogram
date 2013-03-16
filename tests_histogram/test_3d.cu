/*
 *
 *
 *  Created on: 23.2.2012
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
 *
 *
 *
 */

/*
 *
 * Compile with: nvcc -O4 -arch=<your arch> -I../ test_3d.cu -o test3d
 * Needs lemon.mvd to be present in the runtime directory 
 * 
 * Video file header is the following:
 *
 *    All variables little-endian 32-bit two's complement signed integers (ie. normal ints on x86)
 *       - VIDEO_HEADER     (-2777777)
 *       - Header size      (>32)
 *       - width of frame
 *       - Height of frame
 *       - Number of frames
 *       - Format           (unsigned int - Lowest 16 bits reserved (odd) - high 16 bits:
 *                          ( 0 = 24-bit RGB input, 1 = 32-bit RGBA input (ignore Alpha))
 */



#define TESTMAXIDX   64     // Use 64 bins per channel
#define TEST_IS_POW2 1
#define NRUNS 100


#include <string>
#include <iostream>
#include <assert.h>
#include <stdio.h>

#include <cuda_runtime_api.h>
#include "cuda_histogram.h"

#define VIDEO_HEADER    (-2777777)


static bool csv = false;

// Assumes p(x,y,t) is at data[x + y*width + t*width*height]
texture<uchar4, 3, cudaReadModeElementType> tex;

enum getChannels {
    getChannels_rgb = 0,
    getChannels_h2si
};

typedef struct stream_fragment_s
{
    int width;
    int height;
    int frames;
    enum getChannels channels; // (0,1,2) - (r,g,b), (3,4,5) - (h, s, v)
} stream;

static __host__ __device__
void compResults(stream input, uchar4 colors, int* result_index, int* results)
{
    int res;
    switch (input.channels)
    {
        case getChannels_rgb:
            res = colors.x;
            *result_index++ = (res * TESTMAXIDX)>>8;
            res = colors.y;
            *result_index++ = ((res * TESTMAXIDX) >> 8) + TESTMAXIDX;
            res = colors.z;
            *result_index++ = ((res * TESTMAXIDX) >> 8) + 2*TESTMAXIDX;
            *results++ = 1;
            *results++ = 1;
            *results++ = 1;
            break;
        case getChannels_h2si:
        {
            float4 fcolors;
            float x, y;
            float h2, /*c2,*/ i2, s;
            fcolors.x = ((float)colors.x) * 0.00392156862745098f /* / 255.0f */;
            fcolors.y = ((float)colors.y) * 0.00392156862745098f ;
            fcolors.z = ((float)colors.z) * 0.00392156862745098f ;
            float min = fcolors.x;
            if (fcolors.y < min) min = fcolors.y;
            if (fcolors.z < min) min = fcolors.z;
            x = (fcolors.x*2.0f - fcolors.y - fcolors.z)*0.5f;
            y = 0.866f * (fcolors.y - fcolors.z);

            h2 = (3.14159265358979312f + atan2f(y, x)) * 0.159154943091895346f;
            //c2 = sqrt(x*x+y*y);
            i2 = (fcolors.x + fcolors.y + fcolors.z)*0.33333333333f;

            s = (i2 == 0.0f ? 0.0f : 1.0f - min/i2);

            res = (int)(h2 * ((float)TESTMAXIDX - 0.5f));
            *result_index++ = res;
            res = (int)(s * ((float)TESTMAXIDX - 0.5f));
            *result_index++ = res + TESTMAXIDX;
            res = (int)(i2 * ((float)TESTMAXIDX - 0.5f));
            *result_index++ = res + 2*TESTMAXIDX;
            *results++ = 1;
            *results++ = 1;
            *results++ = 1;
            break;
        }
    }
}
// 3 results per input
struct test_xform3
{
  __device__
  void operator() (stream input, int* indices, int* result_index, int* results, int nresults) const {
    int x = indices[0];
    int y = indices[1];
    int t = indices[2];
    float u = ((float)x + 0.5f) / (float)input.width;
    float v = ((float)y + 0.5f) / (float)input.height;
    float t0 = ((float)t + 0.5f) / (float)input.frames;
    uchar4 colors = tex3D(tex, u, v, t0);
    compResults(input, colors, result_index, results);
  }
};
static unsigned int* h_data = NULL;
struct test_xform3_host
{
  void operator() (stream input, int* indices, int* result_index, int* results, int nresults) const {
    int x = indices[0];
    int y = indices[1];
    int t = indices[2];
    // tex3D(tex, u, v, t0);
    uchar4 colors;
    unsigned int packed = h_data[t * input.height * input.width + y * input.width + x];
    colors.x = ((packed >> 0) & 0xff);
    colors.y = ((packed >> 8) & 0xff);
    colors.z = ((packed >> 16) & 0xff);
    colors.w = ((packed >> 24) & 0xff);
    compResults(input, colors, result_index, results);
  }
};

struct test_sumfun2 {
  __device__ __host__
  int operator() (int res1, int res2) const{
    return res1 + res2;
  }
};

static void printresImpl (int* res, int nres, const char* descr)
{
    if (descr)
        printf("\n%s:\n", descr);
    if (csv)
    {
      printf("[\n");
      for (int i = 0; i < nres; i++)
          printf("%d\n", res[i]);
      printf("]\n");
    }
    else
    {
      printf("[ ");
      for (int i = 0; i < nres; i++)
          printf(" %d, ", res[i]);
      printf("]\n");
    }
}

static void printres (int* res, int nres, const char* descr, enum getChannels channels)
{
    if (descr)
        printf("\n%s:\n", descr);
    if (channels == getChannels_rgb)
    {
        printresImpl(res, nres/3, "Red channel");
        printresImpl(&res[nres/3], nres/3, "Green channel");
        printresImpl(&res[2*nres/3], nres/3, "Blue channel");
    }
    else if (channels == getChannels_h2si)
    {
        printresImpl(res, nres/3, "Hue channel");
        printresImpl(&res[nres/3], nres/3, "Saturation channel");
        printresImpl(&res[2*nres/3], nres/3, "Intensity channel");

    }
}

static void testHistogram(stream input, int* xs, int* ends, bool print, bool cpurun, void* nppBuffer, void* nppResBuffer)
{
  int nIndex = TESTMAXIDX;
  test_sumfun2 sumFun;
  test_xform3 xformFun;
  test_xform3_host xformHost;

  //test_indexfun2 indexFun;
  int* tmpres = (int*)malloc(sizeof(int) * nIndex * 3);
  int* cpures = tmpres;
  int zero = 0;
  {
    {
      if (print)
        printf("\nTest reduce_by_key:\n\n");
      memset(tmpres, 0, sizeof(int) * nIndex * 3);
      if (cpurun)
        for (int t = xs[2]; t < ends[2]; t++)
        for (int y = xs[1]; y < ends[1]; y++)
        for (int x = xs[0]; x < ends[0]; x++)
        {
            int index[3];
            int tmp[3];
            int coord[3] = { x, y, t };
            xformHost(input, coord, &index[0], &tmp[0], 3);
            for (int tmpi = 0; tmpi < 3; tmpi++)
            {
                assert(index[tmpi] >= 0);
                assert(index[tmpi] < 3*nIndex);
                cpures[index[tmpi]] = sumFun(cpures[index[tmpi]], tmp[tmpi]);
            }

        }
      if (print && cpurun)
      {
          printres(cpures, nIndex * 3, "CPU results:", input.channels);
      }
    }

    if (!cpurun)
    {
      callHistogramKernelNDim<histogram_atomic_inc, 3, 3>(input, xformFun, sumFun, xs, ends, zero, (int*)nppResBuffer, 3*nIndex, true, 0, nppBuffer);
      cudaMemcpy(tmpres, nppResBuffer, sizeof(int) * TESTMAXIDX * 3, cudaMemcpyDeviceToHost);
    }
    if (print && (!cpurun))
    {
      printres(tmpres, nIndex * 3, "GPU results:", input.channels);
    }

  }
  free(tmpres);
}


void printUsage(void)
{
  printf("\n");
  printf("Test order independent reduce-by-key / histogram algorithm with an image histogram\n\n");
  printf("\tOptions:\n\n");
  printf("\t\t--cpu\t\t\t Run on CPU serially instead of GPU\n");
  printf("\t\t--print\t\t\t Print results of algorithm (check validity)\n");
  printf("\t\t--csv\t\t\t When printing add line-feeds to ease openoffice import...\n");
  printf("\t\t--load <name>\t\t Which media to load\n");
  printf("\t\t--roi <x0 y0 x1 y1>\t Define 2d region of interest \n");
  printf("\t\t--hsv\t\t\t Compute Hue-Saturation-Value histogram instead of RGB \n");
}





static void fillInput(stream& input, const char* filename, int nPixels, int format, int headerSize)
{
  FILE* file = fopen(filename, "rb");
  //texture->dataRGBA8888 = NULL;
  if (!file)
  {
      char* tmp = (char*)malloc(strlen(filename) + 10);
      if (tmp)
      {
          char* ptr = tmp;
          strcpy(ptr, "../");
          ptr += 3;
          strcpy(ptr, filename);
          file = fopen(tmp, "rb");
      }
  }
  // Read
  if (file)
  {
      unsigned int* data = (unsigned int*)h_data;
      if (data)
      {
          int i;
          fseek(file, headerSize, SEEK_SET);
          bool rle = (format & 0x1) != 0;
          format = ((unsigned int)format) >> 16;
          if (rle) printf("RLE-not supported yet - expect garbage...\n");
          int bytesPerPixel = format == 1 ? 4 : 3;
          //input.frames = 0;
          //for (; filesize >= nPixels*bytesPerPixel; filesize -= nPixels*bytesPerPixel)
          for (int t = 0; t < input.frames; t++)
          {
              //input.frames++;
              for (i = 0; i < nPixels; i++)
              {
                  unsigned int raw = 0;
                  int rsize = fread(&raw, bytesPerPixel, 1, file);
                  if (rsize != 1)
                  {
                      printf(
                          "Warning: Unexpected EOF in texture %s at idx %d\n",
                          filename, i);
                      break;
                  }
                  if (bytesPerPixel == 4)
                      *data++ = raw;
                  else
                      *data++ = (raw & 0x00FFFFFF) | ((i & 0xFFu) << 24);
              }
          }
      }
      fclose(file);
  }
}



int main (int argc, char** argv)
{
  int i;

  bool cpu = false;
  bool print = false;
  bool hsv = false;

  int xs[3] = {0};
  int roi[3] = {-1, -1, -1};

  const char* name = "lemon.mvd";

  printUsage();

  for (i = 0; i < argc; i++)
  {
    if (argv[i] && strcmp(argv[i], "--cpu") == 0)
      cpu = true;
    else if (argv[i] && strcmp(argv[i], "--csv") == 0)
      csv = true;
    else if (argv[i] && strcmp(argv[i], "--hsv") == 0)
      hsv = true;
    else if (argv[i] && strcmp(argv[i], "--print") == 0)
      print = true;
    else if (argv[i] && strcmp(argv[i], "--roi") == 0 && argc > i + 4)
    {
        xs[0] = (int)strtol(argv[++i], NULL, 0);
        xs[1] = (int)strtol(argv[++i], NULL, 0);
        roi[0] = (int)strtol(argv[++i], NULL, 0);
        roi[1] = (int)strtol(argv[++i], NULL, 0);
    }
    else if (argv[i] && strcmp(argv[i], "--load") == 0)
    {
      if (argc > i + 1)
        name = argv[i + 1];
    }
  }
  int width = 0;
  int height = 0;
  int format = 0;
  int frames = 0;
  int headerSize = 0;
  {

    int nPixels = 0;
    {
      // Portable way to check filesize with C-apis (of course safe only up to 2GB):
      FILE* file = fopen(name, "rb");
      int error = -1;
      int token = 0;
      if (file)
      {
          // Check header first:
          int check;
          check = fread(&token, 4, 1, file);
          if (token == VIDEO_HEADER)
          {
              check += fread(&headerSize, 4, 1, file);
              check += fread(&width, 4, 1, file);
              check += fread(&height, 4, 1, file);
              check += fread(&frames, 4, 1, file);
              check += fread(&format, 4, 1, file);
              if (check != 6){ 
                  printf("Error reading header - please use funky video-format:\n");
                  return -3;
              }
          }
          if (token != VIDEO_HEADER || headerSize < 32)
          {
              fclose(file);
              printf("Error reading header - please use funky video-format:\n");
              printf("<int token = -2777777>, <int headerSize (>= 32))>,\n\t<int width>, <int height>, <int nFrames>, <int format>, <data> \n");
              printf("formats: <int highest 16 bits: 0-bit use RLE (TODO: Implement & Document),\n");
              printf("int lowest 16 bits: Color-formats: 0 = 32-int ARGB\n");
              return -3;
          }
          error = fseek(file, 0, SEEK_END);
      }
      if (error == 0)
      {
        nPixels = width * height;
      }
      else
      {
        printf("Can't access file: %s, errorcode = %d (man fseek)\n", name, error);
        return error;
      }
    }

    // Allocate keys:
    stream hostIn;
    //int* INPUT = NULL;
    cudaArray *d_volumeArray = 0;

    int* hostINPUT = (int*)malloc(sizeof(int) * nPixels * frames);

    void* nppBuffer = NULL;
    void* nppResBuffer = NULL;

    hostIn.channels = hsv ? getChannels_h2si : getChannels_rgb;
    hostIn.frames = frames;
    hostIn.height = height;
    hostIn.width = width;

    assert(hostINPUT);
    h_data = (unsigned int*)hostINPUT;
    fillInput(hostIn, name, nPixels, format, headerSize);
    if (!cpu)
    {
        // create 3D array
        cudaExtent volumeSize;
        volumeSize.width = width;
        volumeSize.height = height;
        volumeSize.depth = frames;
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize);

        //cudaMalloc(&INPUT, sizeof(int) * nPixels);
        assert(d_volumeArray);

        // copy data to 3D array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr((void*)hostINPUT, volumeSize.width*sizeof(uchar4), volumeSize.width, volumeSize.height);
        copyParams.dstArray = d_volumeArray;
        copyParams.extent   = volumeSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);

        // set texture parameters
        tex.normalized = true;                      // access with normalized texture coordinates
        //tex.filterMode = cudaFilterModeLinear;      // linear interpolation
        tex.filterMode = cudaFilterModePoint;      // Point-sampling - allows easy comparisons with CPU-code
        tex.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
        tex.addressMode[1] = cudaAddressModeWrap;
        tex.addressMode[2] = cudaAddressModeWrap;

        // bind array to 3D texture
        cudaBindTextureToArray(tex, d_volumeArray, channelDesc);

        // Finally allocate result array
        cudaMalloc(&nppResBuffer, sizeof(int) * TESTMAXIDX * 3);
        cudaMemset(nppResBuffer, 0, sizeof(int) * TESTMAXIDX * 3);


    }
    // Create events for timing:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    {
        int zero = 0;
        int tmpbufsize = getHistogramBufSize<histogram_atomic_inc>(zero , (int)(TESTMAXIDX*3));
        cudaMalloc(&nppBuffer, tmpbufsize);
    }

    // Now start timer - we run on stream 0 (default stream):
    cudaEventRecord(start, 0);

    for (i = 0; i < NRUNS; i++)
    {
        if (roi[0] < 0) roi[0] = width;
        if (roi[1] < 0) roi[1] = height;
        roi[2] = frames;
        nPixels = (roi[0] - xs[0]) * (roi[1] - xs[1]) * (roi[2] - xs[2]);
        testHistogram(hostIn, &xs[0], &roi[0], print, cpu, nppBuffer, nppResBuffer);
        // Print only the first time
        print = false;
    }
    {
        float t_ms;
        cudaEventRecord(stop, 0);
        cudaThreadSynchronize();
        cudaEventElapsedTime(&t_ms, start, stop);
        double t = t_ms * 0.001f;
        double GKps = (((double)(3 * nPixels * frames) * (double)NRUNS)) / (t*1.e9);
        printf("Runtime in loops: %fs, Thoughput (Gkeys/s): %3f GK/s \n", t, GKps);
    }
    if (hostINPUT) free(hostINPUT);
    if (nppBuffer) cudaFree(nppBuffer);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaUnbindTexture(tex);
    cudaFreeArray(d_volumeArray);
  }
  return 0;
}

