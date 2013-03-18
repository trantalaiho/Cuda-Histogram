/* A C-program for MT19937: Real number version  (1998/4/6)    */
/*   genrand() generates one pseudorandom real number (double) */
/* which is uniformly distributed on [0,1]-interval, for each  */
/* call. sgenrand(seed) set initial values to the working area */
/* of 624 words. Before genrand(), sgenrand(seed) must be      */
/* called once. (seed is any 32-bit integer except for 0).     */
/* Integer generator is obtained by modifying two lines.       */
/*   Coded by Takuji Nishimura, considering the suggestions by */
/* Topher Cooper and Marc Rieffel in July-Aug. 1997.           */

/* This library is free software; you can redistribute it and/or   */
/* modify it under the terms of the GNU Library General Public     */
/* License as published by the Free Software Foundation; either    */
/* version 2 of the License, or (at your option) any later         */
/* version.                                                        */
/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of  */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.            */
/* See the GNU Library General Public License for more details.    */
/* You should have received a copy of the GNU Library General      */
/* Public License along with this library; if not, write to the    */
/* Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA   */
/* 02111-1307  USA                                                 */

/* Copyright (C) 1997 Makoto Matsumoto and Takuji Nishimura.       */
/* When you use this, send an email to: matumoto@math.keio.ac.jp   */
/* with an appropriate reference to your work.                     */

/* REFERENCE                                                       */
/* M. Matsumoto and T. Nishimura,                                  */
/* "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform  */
/* Pseudo-Random Number Generator",                                */
/* ACM Transactions on Modeling and Computer Simulation,           */
/* Vol. 8, No. 1, January 1998, pp 3--30.                          */

#include<stdlib.h>
#include<stdio.h>

/* Period parameters */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0df   /* constant vector a */
#define UPPER_MASK 0x80000000 /* most significant w-r bits */
#define LOWER_MASK 0x7fffffff /* least significant r bits */

/* Tempering parameters */
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static unsigned int mt[N]; /* the array for the state vector  */
int mersenne_i = -1; /*  < 0 means mt[N] is not initialized */
double mersenne_array[N];

/* initializing the array with a NONZERO seed */
void
seed_mersenne(long seed)
{
  int mti;
  mt[0]= seed & 0xffffffffUL;
  for (mti=1; mti<N; mti++) {
    mt[mti] =
      (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                        */
    /* 2002/01/09 modified by Makoto Matsumoto             */
    mt[mti] &= 0xffffffffUL;
    /* for >32 bit machines */
  }
  mersenne_i = 0;
}


static int s_saveMersenne_i = -1;
static unsigned int save_mt[N]; /* the array for the state vector  */
double save_mersenne_array[N];

void save_mersenne(void)
{
    int i;
    for (i = 0; i < N; i++)
    {
        save_mt[i] = mt[i];
        save_mersenne_array[i] = mersenne_array[i];
    }
    s_saveMersenne_i = mersenne_i;
}
void restore_mersenne(void)
{
    int i;
    for (i = 0; i < N; i++)
    {
        mt[i] = save_mt[i];
        mersenne_array[i] = save_mersenne_array[i];
    }
    mersenne_i = s_saveMersenne_i;

}



double /* generating reals */
/* unsigned int */ /* for integer generation */
mersenne_generate(int *dummy)
{
  register unsigned int y;
  register int kk;
  static unsigned int mag01[2]={0x0, MATRIX_A};
  /* mag01[x] = x * MATRIX_A  for x=0,1 */

  if (mersenne_i < 0) {  /* if sgenrand() has not been called, */
    printf("DUMMY: you did not seed the generator!\n");
    exit(0);
  }

  /* generate N words at one time */

  for (kk=0;kk<N-M;kk++) {
    y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
    mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1];
  }
  for (;kk<N-1;kk++) {
    y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
    mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1];
  }
  y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
  mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1];

  for (kk=0; kk<N; kk++) {
    y = mt[kk];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);
    mersenne_array[kk] = (double)y * 2.3283064365386963e-10;  /* reals: interval [0,1) */
  }

  mersenne_i = N;
  return ( mersenne_array[--mersenne_i] );
    /* return y; */ /* for integer generation */
}

unsigned int /* for integer generation */
mersenne_int(int *dummy)
{
  register unsigned int y;
  register int kk;
  static unsigned int mag01[2]={0x0, MATRIX_A};
  /* mag01[x] = x * MATRIX_A  for x=0,1 */

  if (mersenne_i < 0) {  /* if sgenrand() has not been called, */
    printf("DUMMY: you did not seed the generator!\n");
    exit(0);
  }

  /* generate N words at one time */

  for (kk=0;kk<N-M;kk++) {
    y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
    mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1];
  }
  for (;kk<N-1;kk++) {
    y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
    mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1];
  }
  y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
  mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1];

  for (kk=0; kk<N; kk++) {
    y = mt[kk];
    y ^= TEMPERING_SHIFT_U(y);
    y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
    y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
    y ^= TEMPERING_SHIFT_L(y);
    mersenne_array[kk] = (double)y * 2.3283064365386963e-10;  /* reals: interval [0,1) */
  }

  mersenne_i = N;
  return y; /* for integer generation */
}
