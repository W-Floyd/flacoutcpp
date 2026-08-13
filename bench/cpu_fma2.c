// Same FMA chain, but with explicit AVX2 so the host arm is not
// vectorization-limited: 8 vectors x 4 doubles = 32 independent chains, enough
// to cover FMA latency on both ports.
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NV   8
#define ITER 5000000
static int nthr;
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static void* work(void* arg) {
    __m256d a[NV];
    for (int i = 0; i < NV; i++) a[i] = _mm256_set1_pd(1.0 + i * 1e-3);
    const __m256d m = _mm256_set1_pd(1.0000001), b = _mm256_set1_pd(0.9999999);
    for (long it = 0; it < ITER; it++) {
#pragma GCC unroll 8
        for (int i = 0; i < NV; i++) a[i] = _mm256_fmadd_pd(a[i], m, b);
    }
    __m256d s = a[0];
    for (int i = 1; i < NV; i++) s = _mm256_add_pd(s, a[i]);
    double out[4]; _mm256_storeu_pd(out, s);
    *(double*)arg = out[0] + out[1] + out[2] + out[3];
    return NULL;
}
int main(int argc, char** argv) {
    nthr = argc > 1 ? atoi(argv[1]) : 1;
    pthread_t th[256]; double sink[256];
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        for (int i = 0; i < nthr; i++) pthread_create(&th[i], NULL, work, &sink[i]);
        for (int i = 0; i < nthr; i++) pthread_join(th[i], NULL);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    // 4 doubles per vector, 2 flops per FMA.
    double flops = 2.0 * 4.0 * (double)nthr * NV * (double)ITER;
    printf("fp64 AVX2 %2d thread(s): %8.1f GFLOP/s\n", nthr, flops / best / 1e9);
    return 0;
}
