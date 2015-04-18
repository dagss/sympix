     #include <stdlib.h>
#include <malloc.h>

#ifndef INLINE
# if __STDC_VERSION__ >= 199901L
#  define INLINE inline
# else
#  define INLINE
# endif
#endif

#if (!defined(ENABLE_SSE2) && !defined(ENABLE_AVX))

#if (defined (__AVX__))
#define ENABLE_AVX

#if (defined (__FMA4__))
#define ENABLE_FMA4
#endif

#elif (defined(__SSE2__))
#define ENABLE_SSE2
#else
#error NO SIMD, compile with -march=native
#endif

#endif

#ifdef ENABLE_AVX

#include <immintrin.h>

#ifdef ENABLE_FMA4
#include <x86intrin.h>
#endif

#define VLEN_d 4
#define VLEN_s 8

typedef __m256d vecd;
typedef __m256 vecs;

static INLINE vecd vloadu_d(double *p) { return _mm256_loadu_pd(p); }
static INLINE void vstoreu_d(double *p, vecd v) { _mm256_storeu_pd(p, v); }
static INLINE vecd vadd_d(vecd x, vecd y) { return _mm256_add_pd(x, y); }
static INLINE vecd vsub_d(vecd x, vecd y) { return _mm256_sub_pd(x, y); }
static INLINE vecd vmul_d(vecd x, vecd y) { return _mm256_mul_pd(x, y); }
static INLINE vecd vcast_d(double d) { return _mm256_set_pd(d, d, d, d); }
static INLINE vecd vbroadcast_d(double *p) {
    return _mm256_broadcast_sd(p);
}

static INLINE vecs vloadu_s(float *p) { return _mm256_loadu_ps(p); }
static INLINE void vstoreu_s(float *p, vecs v) { _mm256_storeu_ps(p, v); }
static INLINE vecs vadd_s(vecs x, vecs y) { return _mm256_add_ps(x, y); }
static INLINE vecs vsub_s(vecs x, vecs y) { return _mm256_sub_ps(x, y); }
static INLINE vecs vmul_s(vecs x, vecs y) { return _mm256_mul_ps(x, y); }
static INLINE vecs vcast_s(float x) { return _mm256_set_ps(x, x, x, x, x, x, x, x); }
static INLINE vecs vbroadcast_s(float *p) {
    return _mm256_broadcast_ss(p);
}



#ifdef ENABLE_FMA4
static INLINE vecd vmacc_d(vecd x, vecd y, vecd z) {
    return _mm256_macc_pd(x, y, z);
}
static INLINE vecs vmacc_s(vecs x, vecs y, vecs z) {
    return _mm256_macc_ps(x, y, z);
}
#else
static INLINE vecd vmacc_d(vecd x, vecd y, vecd z) {
    return vadd_d(vmul_d(x, y), z);
}
static INLINE vecs vmacc_s(vecs x, vecs y, vecs z) {
    return vadd_s(vmul_s(x, y), z);
}
#endif

#endif


#ifdef ENABLE_SSE2

#include <immintrin.h>

#define VLEN_d 2
#define VLEN_s 4

typedef __m128d vecd;
typedef __m128 vecs;

static INLINE vecd vloadu_d(double *p) { return _mm_loadu_pd(p); }
static INLINE void vstoreu_d(double *p, vecd v) { _mm_storeu_pd(p, v); }
static INLINE vecd vadd_d(vecd x, vecd y) { return _mm_add_pd(x, y); }
static INLINE vecd vsub_d(vecd x, vecd y) { return _mm_sub_pd(x, y); }
static INLINE vecd vmul_d(vecd x, vecd y) { return _mm_mul_pd(x, y); }
static INLINE vecd vmacc_d(vecd x, vecd y, vecd z) {
    return vadd_d(vmul_d(x, y), z);
}
static INLINE vecd vcast_d(double d) { return _mm_set_pd(d, d); }
static INLINE vecd vbroadcast_d(double *p) {
    return _mm_set_pd(*p, *p);
}

static INLINE vecs vloadu_s(float *p) { return _mm_loadu_ps(p); }
static INLINE void vstoreu_s(float *p, vecs v) { _mm_storeu_ps(p, v); }
static INLINE vecs vadd_s(vecs x, vecs y) { return _mm_add_ps(x, y); }
static INLINE vecs vsub_s(vecs x, vecs y) { return _mm_sub_ps(x, y); }
static INLINE vecs vmul_s(vecs x, vecs y) { return _mm_mul_ps(x, y); }
static INLINE vecs vmacc_s(vecs x, vecs y, vecs z) {
    return vadd_s(vmul_s(x, y), z);
}
static INLINE vecs vcast_s(float x) { return _mm_set_ps(x, x, x, x); }
static INLINE vecs vbroadcast_s(float *p) {
    return _mm_set_ps(*p, *p, *p, *p);
}



#endif



static void legendre_transform_vec1_d(double *recfacs, double *bl, size_t lmax,
                                               double xarr[(1) * VLEN_d],
                                               double out[(1) * VLEN_d]) {
    
    vecd P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecd W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_d(xarr + 0 * VLEN_d);
    Pm1_0 = vcast_d(1.0);
    P_0 = x0;
    b = vbroadcast_d(bl);
    y0 = vmul_d(Pm1_0, b);
    
    
    b = vbroadcast_d(bl + 1);
    
    y0 = vmacc_d(P_0, b, y0);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_d(bl + l);
        R = vbroadcast_d(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_d(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_0);
        P_0 = vmacc_d(W2, R, W1);
        y0 = vmacc_d(P_0, b, y0);
        

    }
    
    vstoreu_d(out + 0 * VLEN_d, y0);
    
}

static void legendre_transform_vec2_d(double *recfacs, double *bl, size_t lmax,
                                               double xarr[(2) * VLEN_d],
                                               double out[(2) * VLEN_d]) {
    
    vecd P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecd P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecd W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_d(xarr + 0 * VLEN_d);
    Pm1_0 = vcast_d(1.0);
    P_0 = x0;
    b = vbroadcast_d(bl);
    y0 = vmul_d(Pm1_0, b);
    
    x1 = vloadu_d(xarr + 1 * VLEN_d);
    Pm1_1 = vcast_d(1.0);
    P_1 = x1;
    b = vbroadcast_d(bl);
    y1 = vmul_d(Pm1_1, b);
    
    
    b = vbroadcast_d(bl + 1);
    
    y0 = vmacc_d(P_0, b, y0);
    
    y1 = vmacc_d(P_1, b, y1);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_d(bl + l);
        R = vbroadcast_d(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_d(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_0);
        P_0 = vmacc_d(W2, R, W1);
        y0 = vmacc_d(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_d(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_1);
        P_1 = vmacc_d(W2, R, W1);
        y1 = vmacc_d(P_1, b, y1);
        

    }
    
    vstoreu_d(out + 0 * VLEN_d, y0);
    
    vstoreu_d(out + 1 * VLEN_d, y1);
    
}

static void legendre_transform_vec3_d(double *recfacs, double *bl, size_t lmax,
                                               double xarr[(3) * VLEN_d],
                                               double out[(3) * VLEN_d]) {
    
    vecd P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecd P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecd P_2, Pm1_2, Pm2_2, x2, y2;
    
    vecd W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_d(xarr + 0 * VLEN_d);
    Pm1_0 = vcast_d(1.0);
    P_0 = x0;
    b = vbroadcast_d(bl);
    y0 = vmul_d(Pm1_0, b);
    
    x1 = vloadu_d(xarr + 1 * VLEN_d);
    Pm1_1 = vcast_d(1.0);
    P_1 = x1;
    b = vbroadcast_d(bl);
    y1 = vmul_d(Pm1_1, b);
    
    x2 = vloadu_d(xarr + 2 * VLEN_d);
    Pm1_2 = vcast_d(1.0);
    P_2 = x2;
    b = vbroadcast_d(bl);
    y2 = vmul_d(Pm1_2, b);
    
    
    b = vbroadcast_d(bl + 1);
    
    y0 = vmacc_d(P_0, b, y0);
    
    y1 = vmacc_d(P_1, b, y1);
    
    y2 = vmacc_d(P_2, b, y2);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_d(bl + l);
        R = vbroadcast_d(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_d(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_0);
        P_0 = vmacc_d(W2, R, W1);
        y0 = vmacc_d(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_d(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_1);
        P_1 = vmacc_d(W2, R, W1);
        y1 = vmacc_d(P_1, b, y1);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_d(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_2);
        P_2 = vmacc_d(W2, R, W1);
        y2 = vmacc_d(P_2, b, y2);
        

    }
    
    vstoreu_d(out + 0 * VLEN_d, y0);
    
    vstoreu_d(out + 1 * VLEN_d, y1);
    
    vstoreu_d(out + 2 * VLEN_d, y2);
    
}

static void legendre_transform_vec4_d(double *recfacs, double *bl, size_t lmax,
                                               double xarr[(4) * VLEN_d],
                                               double out[(4) * VLEN_d]) {
    
    vecd P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecd P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecd P_2, Pm1_2, Pm2_2, x2, y2;
    
    vecd P_3, Pm1_3, Pm2_3, x3, y3;
    
    vecd W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_d(xarr + 0 * VLEN_d);
    Pm1_0 = vcast_d(1.0);
    P_0 = x0;
    b = vbroadcast_d(bl);
    y0 = vmul_d(Pm1_0, b);
    
    x1 = vloadu_d(xarr + 1 * VLEN_d);
    Pm1_1 = vcast_d(1.0);
    P_1 = x1;
    b = vbroadcast_d(bl);
    y1 = vmul_d(Pm1_1, b);
    
    x2 = vloadu_d(xarr + 2 * VLEN_d);
    Pm1_2 = vcast_d(1.0);
    P_2 = x2;
    b = vbroadcast_d(bl);
    y2 = vmul_d(Pm1_2, b);
    
    x3 = vloadu_d(xarr + 3 * VLEN_d);
    Pm1_3 = vcast_d(1.0);
    P_3 = x3;
    b = vbroadcast_d(bl);
    y3 = vmul_d(Pm1_3, b);
    
    
    b = vbroadcast_d(bl + 1);
    
    y0 = vmacc_d(P_0, b, y0);
    
    y1 = vmacc_d(P_1, b, y1);
    
    y2 = vmacc_d(P_2, b, y2);
    
    y3 = vmacc_d(P_3, b, y3);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_d(bl + l);
        R = vbroadcast_d(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_d(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_0);
        P_0 = vmacc_d(W2, R, W1);
        y0 = vmacc_d(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_d(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_1);
        P_1 = vmacc_d(W2, R, W1);
        y1 = vmacc_d(P_1, b, y1);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_d(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_2);
        P_2 = vmacc_d(W2, R, W1);
        y2 = vmacc_d(P_2, b, y2);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul_d(x3, Pm1_3);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_3);
        P_3 = vmacc_d(W2, R, W1);
        y3 = vmacc_d(P_3, b, y3);
        

    }
    
    vstoreu_d(out + 0 * VLEN_d, y0);
    
    vstoreu_d(out + 1 * VLEN_d, y1);
    
    vstoreu_d(out + 2 * VLEN_d, y2);
    
    vstoreu_d(out + 3 * VLEN_d, y3);
    
}

static void legendre_transform_vec5_d(double *recfacs, double *bl, size_t lmax,
                                               double xarr[(5) * VLEN_d],
                                               double out[(5) * VLEN_d]) {
    
    vecd P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecd P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecd P_2, Pm1_2, Pm2_2, x2, y2;
    
    vecd P_3, Pm1_3, Pm2_3, x3, y3;
    
    vecd P_4, Pm1_4, Pm2_4, x4, y4;
    
    vecd W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_d(xarr + 0 * VLEN_d);
    Pm1_0 = vcast_d(1.0);
    P_0 = x0;
    b = vbroadcast_d(bl);
    y0 = vmul_d(Pm1_0, b);
    
    x1 = vloadu_d(xarr + 1 * VLEN_d);
    Pm1_1 = vcast_d(1.0);
    P_1 = x1;
    b = vbroadcast_d(bl);
    y1 = vmul_d(Pm1_1, b);
    
    x2 = vloadu_d(xarr + 2 * VLEN_d);
    Pm1_2 = vcast_d(1.0);
    P_2 = x2;
    b = vbroadcast_d(bl);
    y2 = vmul_d(Pm1_2, b);
    
    x3 = vloadu_d(xarr + 3 * VLEN_d);
    Pm1_3 = vcast_d(1.0);
    P_3 = x3;
    b = vbroadcast_d(bl);
    y3 = vmul_d(Pm1_3, b);
    
    x4 = vloadu_d(xarr + 4 * VLEN_d);
    Pm1_4 = vcast_d(1.0);
    P_4 = x4;
    b = vbroadcast_d(bl);
    y4 = vmul_d(Pm1_4, b);
    
    
    b = vbroadcast_d(bl + 1);
    
    y0 = vmacc_d(P_0, b, y0);
    
    y1 = vmacc_d(P_1, b, y1);
    
    y2 = vmacc_d(P_2, b, y2);
    
    y3 = vmacc_d(P_3, b, y3);
    
    y4 = vmacc_d(P_4, b, y4);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_d(bl + l);
        R = vbroadcast_d(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_d(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_0);
        P_0 = vmacc_d(W2, R, W1);
        y0 = vmacc_d(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_d(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_1);
        P_1 = vmacc_d(W2, R, W1);
        y1 = vmacc_d(P_1, b, y1);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_d(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_2);
        P_2 = vmacc_d(W2, R, W1);
        y2 = vmacc_d(P_2, b, y2);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul_d(x3, Pm1_3);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_3);
        P_3 = vmacc_d(W2, R, W1);
        y3 = vmacc_d(P_3, b, y3);
        
        Pm2_4 = Pm1_4; Pm1_4 = P_4;
        W1 = vmul_d(x4, Pm1_4);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_4);
        P_4 = vmacc_d(W2, R, W1);
        y4 = vmacc_d(P_4, b, y4);
        

    }
    
    vstoreu_d(out + 0 * VLEN_d, y0);
    
    vstoreu_d(out + 1 * VLEN_d, y1);
    
    vstoreu_d(out + 2 * VLEN_d, y2);
    
    vstoreu_d(out + 3 * VLEN_d, y3);
    
    vstoreu_d(out + 4 * VLEN_d, y4);
    
}

static void legendre_transform_vec6_d(double *recfacs, double *bl, size_t lmax,
                                               double xarr[(6) * VLEN_d],
                                               double out[(6) * VLEN_d]) {
    
    vecd P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecd P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecd P_2, Pm1_2, Pm2_2, x2, y2;
    
    vecd P_3, Pm1_3, Pm2_3, x3, y3;
    
    vecd P_4, Pm1_4, Pm2_4, x4, y4;
    
    vecd P_5, Pm1_5, Pm2_5, x5, y5;
    
    vecd W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_d(xarr + 0 * VLEN_d);
    Pm1_0 = vcast_d(1.0);
    P_0 = x0;
    b = vbroadcast_d(bl);
    y0 = vmul_d(Pm1_0, b);
    
    x1 = vloadu_d(xarr + 1 * VLEN_d);
    Pm1_1 = vcast_d(1.0);
    P_1 = x1;
    b = vbroadcast_d(bl);
    y1 = vmul_d(Pm1_1, b);
    
    x2 = vloadu_d(xarr + 2 * VLEN_d);
    Pm1_2 = vcast_d(1.0);
    P_2 = x2;
    b = vbroadcast_d(bl);
    y2 = vmul_d(Pm1_2, b);
    
    x3 = vloadu_d(xarr + 3 * VLEN_d);
    Pm1_3 = vcast_d(1.0);
    P_3 = x3;
    b = vbroadcast_d(bl);
    y3 = vmul_d(Pm1_3, b);
    
    x4 = vloadu_d(xarr + 4 * VLEN_d);
    Pm1_4 = vcast_d(1.0);
    P_4 = x4;
    b = vbroadcast_d(bl);
    y4 = vmul_d(Pm1_4, b);
    
    x5 = vloadu_d(xarr + 5 * VLEN_d);
    Pm1_5 = vcast_d(1.0);
    P_5 = x5;
    b = vbroadcast_d(bl);
    y5 = vmul_d(Pm1_5, b);
    
    
    b = vbroadcast_d(bl + 1);
    
    y0 = vmacc_d(P_0, b, y0);
    
    y1 = vmacc_d(P_1, b, y1);
    
    y2 = vmacc_d(P_2, b, y2);
    
    y3 = vmacc_d(P_3, b, y3);
    
    y4 = vmacc_d(P_4, b, y4);
    
    y5 = vmacc_d(P_5, b, y5);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_d(bl + l);
        R = vbroadcast_d(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_d(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_0);
        P_0 = vmacc_d(W2, R, W1);
        y0 = vmacc_d(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_d(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_1);
        P_1 = vmacc_d(W2, R, W1);
        y1 = vmacc_d(P_1, b, y1);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_d(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_2);
        P_2 = vmacc_d(W2, R, W1);
        y2 = vmacc_d(P_2, b, y2);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul_d(x3, Pm1_3);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_3);
        P_3 = vmacc_d(W2, R, W1);
        y3 = vmacc_d(P_3, b, y3);
        
        Pm2_4 = Pm1_4; Pm1_4 = P_4;
        W1 = vmul_d(x4, Pm1_4);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_4);
        P_4 = vmacc_d(W2, R, W1);
        y4 = vmacc_d(P_4, b, y4);
        
        Pm2_5 = Pm1_5; Pm1_5 = P_5;
        W1 = vmul_d(x5, Pm1_5);
        W2 = W1;
        W2 = vsub_d(W2, Pm2_5);
        P_5 = vmacc_d(W2, R, W1);
        y5 = vmacc_d(P_5, b, y5);
        

    }
    
    vstoreu_d(out + 0 * VLEN_d, y0);
    
    vstoreu_d(out + 1 * VLEN_d, y1);
    
    vstoreu_d(out + 2 * VLEN_d, y2);
    
    vstoreu_d(out + 3 * VLEN_d, y3);
    
    vstoreu_d(out + 4 * VLEN_d, y4);
    
    vstoreu_d(out + 5 * VLEN_d, y5);
    
}



static void legendre_transform_vec1_s(float *recfacs, float *bl, size_t lmax,
                                               float xarr[(1) * VLEN_s],
                                               float out[(1) * VLEN_s]) {
    
    vecs P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecs W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vcast_s(1.0);
    P_0 = x0;
    b = vbroadcast_s(bl);
    y0 = vmul_s(Pm1_0, b);
    
    
    b = vbroadcast_s(bl + 1);
    
    y0 = vmacc_s(P_0, b, y0);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_s(bl + l);
        R = vbroadcast_s(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = vmacc_s(W2, R, W1);
        y0 = vmacc_s(P_0, b, y0);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
}

static void legendre_transform_vec2_s(float *recfacs, float *bl, size_t lmax,
                                               float xarr[(2) * VLEN_s],
                                               float out[(2) * VLEN_s]) {
    
    vecs P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecs P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecs W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vcast_s(1.0);
    P_0 = x0;
    b = vbroadcast_s(bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vcast_s(1.0);
    P_1 = x1;
    b = vbroadcast_s(bl);
    y1 = vmul_s(Pm1_1, b);
    
    
    b = vbroadcast_s(bl + 1);
    
    y0 = vmacc_s(P_0, b, y0);
    
    y1 = vmacc_s(P_1, b, y1);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_s(bl + l);
        R = vbroadcast_s(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = vmacc_s(W2, R, W1);
        y0 = vmacc_s(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = vmacc_s(W2, R, W1);
        y1 = vmacc_s(P_1, b, y1);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
}

static void legendre_transform_vec3_s(float *recfacs, float *bl, size_t lmax,
                                               float xarr[(3) * VLEN_s],
                                               float out[(3) * VLEN_s]) {
    
    vecs P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecs P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecs P_2, Pm1_2, Pm2_2, x2, y2;
    
    vecs W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vcast_s(1.0);
    P_0 = x0;
    b = vbroadcast_s(bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vcast_s(1.0);
    P_1 = x1;
    b = vbroadcast_s(bl);
    y1 = vmul_s(Pm1_1, b);
    
    x2 = vloadu_s(xarr + 2 * VLEN_s);
    Pm1_2 = vcast_s(1.0);
    P_2 = x2;
    b = vbroadcast_s(bl);
    y2 = vmul_s(Pm1_2, b);
    
    
    b = vbroadcast_s(bl + 1);
    
    y0 = vmacc_s(P_0, b, y0);
    
    y1 = vmacc_s(P_1, b, y1);
    
    y2 = vmacc_s(P_2, b, y2);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_s(bl + l);
        R = vbroadcast_s(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = vmacc_s(W2, R, W1);
        y0 = vmacc_s(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = vmacc_s(W2, R, W1);
        y1 = vmacc_s(P_1, b, y1);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_s(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_2);
        P_2 = vmacc_s(W2, R, W1);
        y2 = vmacc_s(P_2, b, y2);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
    vstoreu_s(out + 2 * VLEN_s, y2);
    
}

static void legendre_transform_vec4_s(float *recfacs, float *bl, size_t lmax,
                                               float xarr[(4) * VLEN_s],
                                               float out[(4) * VLEN_s]) {
    
    vecs P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecs P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecs P_2, Pm1_2, Pm2_2, x2, y2;
    
    vecs P_3, Pm1_3, Pm2_3, x3, y3;
    
    vecs W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vcast_s(1.0);
    P_0 = x0;
    b = vbroadcast_s(bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vcast_s(1.0);
    P_1 = x1;
    b = vbroadcast_s(bl);
    y1 = vmul_s(Pm1_1, b);
    
    x2 = vloadu_s(xarr + 2 * VLEN_s);
    Pm1_2 = vcast_s(1.0);
    P_2 = x2;
    b = vbroadcast_s(bl);
    y2 = vmul_s(Pm1_2, b);
    
    x3 = vloadu_s(xarr + 3 * VLEN_s);
    Pm1_3 = vcast_s(1.0);
    P_3 = x3;
    b = vbroadcast_s(bl);
    y3 = vmul_s(Pm1_3, b);
    
    
    b = vbroadcast_s(bl + 1);
    
    y0 = vmacc_s(P_0, b, y0);
    
    y1 = vmacc_s(P_1, b, y1);
    
    y2 = vmacc_s(P_2, b, y2);
    
    y3 = vmacc_s(P_3, b, y3);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_s(bl + l);
        R = vbroadcast_s(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = vmacc_s(W2, R, W1);
        y0 = vmacc_s(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = vmacc_s(W2, R, W1);
        y1 = vmacc_s(P_1, b, y1);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_s(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_2);
        P_2 = vmacc_s(W2, R, W1);
        y2 = vmacc_s(P_2, b, y2);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul_s(x3, Pm1_3);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_3);
        P_3 = vmacc_s(W2, R, W1);
        y3 = vmacc_s(P_3, b, y3);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
    vstoreu_s(out + 2 * VLEN_s, y2);
    
    vstoreu_s(out + 3 * VLEN_s, y3);
    
}

static void legendre_transform_vec5_s(float *recfacs, float *bl, size_t lmax,
                                               float xarr[(5) * VLEN_s],
                                               float out[(5) * VLEN_s]) {
    
    vecs P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecs P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecs P_2, Pm1_2, Pm2_2, x2, y2;
    
    vecs P_3, Pm1_3, Pm2_3, x3, y3;
    
    vecs P_4, Pm1_4, Pm2_4, x4, y4;
    
    vecs W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vcast_s(1.0);
    P_0 = x0;
    b = vbroadcast_s(bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vcast_s(1.0);
    P_1 = x1;
    b = vbroadcast_s(bl);
    y1 = vmul_s(Pm1_1, b);
    
    x2 = vloadu_s(xarr + 2 * VLEN_s);
    Pm1_2 = vcast_s(1.0);
    P_2 = x2;
    b = vbroadcast_s(bl);
    y2 = vmul_s(Pm1_2, b);
    
    x3 = vloadu_s(xarr + 3 * VLEN_s);
    Pm1_3 = vcast_s(1.0);
    P_3 = x3;
    b = vbroadcast_s(bl);
    y3 = vmul_s(Pm1_3, b);
    
    x4 = vloadu_s(xarr + 4 * VLEN_s);
    Pm1_4 = vcast_s(1.0);
    P_4 = x4;
    b = vbroadcast_s(bl);
    y4 = vmul_s(Pm1_4, b);
    
    
    b = vbroadcast_s(bl + 1);
    
    y0 = vmacc_s(P_0, b, y0);
    
    y1 = vmacc_s(P_1, b, y1);
    
    y2 = vmacc_s(P_2, b, y2);
    
    y3 = vmacc_s(P_3, b, y3);
    
    y4 = vmacc_s(P_4, b, y4);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_s(bl + l);
        R = vbroadcast_s(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = vmacc_s(W2, R, W1);
        y0 = vmacc_s(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = vmacc_s(W2, R, W1);
        y1 = vmacc_s(P_1, b, y1);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_s(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_2);
        P_2 = vmacc_s(W2, R, W1);
        y2 = vmacc_s(P_2, b, y2);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul_s(x3, Pm1_3);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_3);
        P_3 = vmacc_s(W2, R, W1);
        y3 = vmacc_s(P_3, b, y3);
        
        Pm2_4 = Pm1_4; Pm1_4 = P_4;
        W1 = vmul_s(x4, Pm1_4);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_4);
        P_4 = vmacc_s(W2, R, W1);
        y4 = vmacc_s(P_4, b, y4);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
    vstoreu_s(out + 2 * VLEN_s, y2);
    
    vstoreu_s(out + 3 * VLEN_s, y3);
    
    vstoreu_s(out + 4 * VLEN_s, y4);
    
}

static void legendre_transform_vec6_s(float *recfacs, float *bl, size_t lmax,
                                               float xarr[(6) * VLEN_s],
                                               float out[(6) * VLEN_s]) {
    
    vecs P_0, Pm1_0, Pm2_0, x0, y0;
    
    vecs P_1, Pm1_1, Pm2_1, x1, y1;
    
    vecs P_2, Pm1_2, Pm2_2, x2, y2;
    
    vecs P_3, Pm1_3, Pm2_3, x3, y3;
    
    vecs P_4, Pm1_4, Pm2_4, x4, y4;
    
    vecs P_5, Pm1_5, Pm2_5, x5, y5;
    
    vecs W1, W2, b, R;
    ssize_t l;

    
    x0 = vloadu_s(xarr + 0 * VLEN_s);
    Pm1_0 = vcast_s(1.0);
    P_0 = x0;
    b = vbroadcast_s(bl);
    y0 = vmul_s(Pm1_0, b);
    
    x1 = vloadu_s(xarr + 1 * VLEN_s);
    Pm1_1 = vcast_s(1.0);
    P_1 = x1;
    b = vbroadcast_s(bl);
    y1 = vmul_s(Pm1_1, b);
    
    x2 = vloadu_s(xarr + 2 * VLEN_s);
    Pm1_2 = vcast_s(1.0);
    P_2 = x2;
    b = vbroadcast_s(bl);
    y2 = vmul_s(Pm1_2, b);
    
    x3 = vloadu_s(xarr + 3 * VLEN_s);
    Pm1_3 = vcast_s(1.0);
    P_3 = x3;
    b = vbroadcast_s(bl);
    y3 = vmul_s(Pm1_3, b);
    
    x4 = vloadu_s(xarr + 4 * VLEN_s);
    Pm1_4 = vcast_s(1.0);
    P_4 = x4;
    b = vbroadcast_s(bl);
    y4 = vmul_s(Pm1_4, b);
    
    x5 = vloadu_s(xarr + 5 * VLEN_s);
    Pm1_5 = vcast_s(1.0);
    P_5 = x5;
    b = vbroadcast_s(bl);
    y5 = vmul_s(Pm1_5, b);
    
    
    b = vbroadcast_s(bl + 1);
    
    y0 = vmacc_s(P_0, b, y0);
    
    y1 = vmacc_s(P_1, b, y1);
    
    y2 = vmacc_s(P_2, b, y2);
    
    y3 = vmacc_s(P_3, b, y3);
    
    y4 = vmacc_s(P_4, b, y4);
    
    y5 = vmacc_s(P_5, b, y5);
    

    for (l = 2; l <= lmax; ++l) {
        b = vbroadcast_s(bl + l);
        R = vbroadcast_s(recfacs + l);
        
        /* 
           P = x * Pm1 + recfacs[l] * (x * Pm1 - Pm2)
        */
        
        Pm2_0 = Pm1_0; Pm1_0 = P_0;
        W1 = vmul_s(x0, Pm1_0);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_0);
        P_0 = vmacc_s(W2, R, W1);
        y0 = vmacc_s(P_0, b, y0);
        
        Pm2_1 = Pm1_1; Pm1_1 = P_1;
        W1 = vmul_s(x1, Pm1_1);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_1);
        P_1 = vmacc_s(W2, R, W1);
        y1 = vmacc_s(P_1, b, y1);
        
        Pm2_2 = Pm1_2; Pm1_2 = P_2;
        W1 = vmul_s(x2, Pm1_2);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_2);
        P_2 = vmacc_s(W2, R, W1);
        y2 = vmacc_s(P_2, b, y2);
        
        Pm2_3 = Pm1_3; Pm1_3 = P_3;
        W1 = vmul_s(x3, Pm1_3);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_3);
        P_3 = vmacc_s(W2, R, W1);
        y3 = vmacc_s(P_3, b, y3);
        
        Pm2_4 = Pm1_4; Pm1_4 = P_4;
        W1 = vmul_s(x4, Pm1_4);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_4);
        P_4 = vmacc_s(W2, R, W1);
        y4 = vmacc_s(P_4, b, y4);
        
        Pm2_5 = Pm1_5; Pm1_5 = P_5;
        W1 = vmul_s(x5, Pm1_5);
        W2 = W1;
        W2 = vsub_s(W2, Pm2_5);
        P_5 = vmacc_s(W2, R, W1);
        y5 = vmacc_s(P_5, b, y5);
        

    }
    
    vstoreu_s(out + 0 * VLEN_s, y0);
    
    vstoreu_s(out + 1 * VLEN_s, y1);
    
    vstoreu_s(out + 2 * VLEN_s, y2);
    
    vstoreu_s(out + 3 * VLEN_s, y3);
    
    vstoreu_s(out + 4 * VLEN_s, y4);
    
    vstoreu_s(out + 5 * VLEN_s, y5);
    
}





void legendre_transform_recfac_d(double *r, size_t lmax) {
    /* (l - 1) / l, for l >= 2 */
    size_t l;
    r[0] = 0;
    r[1] = 1;
    for (l = 2; l <= lmax; ++l) {
        r[l] = (double)(l - 1) / (double)l;
    }
}

void legendre_transform_recfac_s(float *r, size_t lmax) {
    /* (l - 1) / l, for l >= 2 */
    size_t l;
    r[0] = 0;
    r[1] = 1;
    for (l = 2; l <= lmax; ++l) {
        r[l] = (float)(l - 1) / (float)l;
    }
}


/*
  Compute sum_l b_l P_l(x_i) for all i. 
 */



#define CS 4
#define LEN_d (CS * VLEN_d)
#define LEN_s (CS * VLEN_s)


void legendre_transform_d(double *bl,
                              double *recfac,
                              size_t lmax,
                              double *x, double *out, size_t nx) {
    double xchunk[LEN_d], outchunk[LEN_d];
    int compute_recfac;
    size_t i, j, len;

    compute_recfac = (recfac == NULL);
    if (compute_recfac) {
        recfac = memalign(16, sizeof(double) * (lmax + 1));
        legendre_transform_recfac_d(recfac, lmax);
    }

    for (j = 0; j != LEN_d; ++j) xchunk[j] = 0;

    for (i = 0; i < nx; i += LEN_d) {
        len = (i + (LEN_d) <= nx) ? (LEN_d) : (nx - i);
        for (j = 0; j != len; ++j) xchunk[j] = x[i + j];
        switch ((len + VLEN_d - 1) / VLEN_d) {
          case 6: legendre_transform_vec6_d(recfac, bl, lmax, xchunk, outchunk); break;
          case 5: legendre_transform_vec5_d(recfac, bl, lmax, xchunk, outchunk); break;
          case 4: legendre_transform_vec4_d(recfac, bl, lmax, xchunk, outchunk); break;
          case 3: legendre_transform_vec3_d(recfac, bl, lmax, xchunk, outchunk); break;
          case 2: legendre_transform_vec2_d(recfac, bl, lmax, xchunk, outchunk); break;
          case 1:
          case 0:
              legendre_transform_vec1_d(recfac, bl, lmax, xchunk, outchunk); break;
        }
        for (j = 0; j != len; ++j) out[i + j] = outchunk[j];
    }
    if (compute_recfac) {
        free(recfac);
    }
}

void legendre_transform_s(float *bl,
                              float *recfac,
                              size_t lmax,
                              float *x, float *out, size_t nx) {
    float xchunk[LEN_s], outchunk[LEN_s];
    int compute_recfac;
    size_t i, j, len;

    compute_recfac = (recfac == NULL);
    if (compute_recfac) {
        recfac = memalign(16, sizeof(float) * (lmax + 1));
        legendre_transform_recfac_s(recfac, lmax);
    }

    for (j = 0; j != LEN_s; ++j) xchunk[j] = 0;

    for (i = 0; i < nx; i += LEN_s) {
        len = (i + (LEN_s) <= nx) ? (LEN_s) : (nx - i);
        for (j = 0; j != len; ++j) xchunk[j] = x[i + j];
        switch ((len + VLEN_s - 1) / VLEN_s) {
          case 6: legendre_transform_vec6_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 5: legendre_transform_vec5_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 4: legendre_transform_vec4_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 3: legendre_transform_vec3_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 2: legendre_transform_vec2_s(recfac, bl, lmax, xchunk, outchunk); break;
          case 1:
          case 0:
              legendre_transform_vec1_s(recfac, bl, lmax, xchunk, outchunk); break;
        }
        for (j = 0; j != len; ++j) out[i + j] = outchunk[j];
    }
    if (compute_recfac) {
        free(recfac);
    }
}

