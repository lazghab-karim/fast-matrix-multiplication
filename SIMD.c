#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>

#if defined(_MSC_VER) || defined(_WIN32)
#  include <malloc.h>
#  define ALIGN_ALLOC(sz, aln) _aligned_malloc((sz), (aln))
#  define ALIGN_FREE(ptr) _aligned_free(ptr)
#else
#  define ALIGN_ALLOC(sz, aln) ({ \
      void* p = NULL; \
      if (posix_memalign(&p, (aln), (sz)) != 0) p = NULL; \
      p; })
#  define ALIGN_FREE(ptr) free(ptr)
#endif
static inline void sgemm_scalar_block(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int i0, int imax, int j0, int jmax)
{
    for (int i = i0; i < imax; ++i) {
        for (int j = j0; j < jmax; ++j) {
            float sum = 0.0f;
            const float* arow = A + (size_t)i * lda;
            const float* bcol = B + (size_t)j;
            for (int k = 0; k < K; ++k)
                sum += arow[k] * bcol[(size_t)k * ldb];
            C[(size_t)i * ldc + j] += sum;
        }
    }
}

#if defined(__AVX2__)
static inline void sgemm_kernel_8x8(
    int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc,
    int i0, int j0)
{
    __m256 c0 = _mm256_setzero_ps();
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    __m256 c5 = _mm256_setzero_ps();
    __m256 c6 = _mm256_setzero_ps();
    __m256 c7 = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        const float* brow = B + (size_t)k * ldb + j0;
        __m256 b = _mm256_loadu_ps(brow);
        const float* ak = A + (size_t)i0 * lda + k;
        __m256 a0 = _mm256_broadcast_ss(ak + 0*lda);
        __m256 a1 = _mm256_broadcast_ss(ak + 1*lda);
        __m256 a2 = _mm256_broadcast_ss(ak + 2*lda);
        __m256 a3 = _mm256_broadcast_ss(ak + 3*lda);
        __m256 a4 = _mm256_broadcast_ss(ak + 4*lda);
        __m256 a5 = _mm256_broadcast_ss(ak + 5*lda);
        __m256 a6 = _mm256_broadcast_ss(ak + 6*lda);
        __m256 a7 = _mm256_broadcast_ss(ak + 7*lda);
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
        c4 = _mm256_fmadd_ps(a4, b, c4);
        c5 = _mm256_fmadd_ps(a5, b, c5);
        c6 = _mm256_fmadd_ps(a6, b, c6);
        c7 = _mm256_fmadd_ps(a7, b, c7);
    }
    _mm256_storeu_ps(C + (size_t)(i0+0) * ldc + j0, c0);
    _mm256_storeu_ps(C + (size_t)(i0+1) * ldc + j0, c1);
    _mm256_storeu_ps(C + (size_t)(i0+2) * ldc + j0, c2);
    _mm256_storeu_ps(C + (size_t)(i0+3) * ldc + j0, c3);
    _mm256_storeu_ps(C + (size_t)(i0+4) * ldc + j0, c4);
    _mm256_storeu_ps(C + (size_t)(i0+5) * ldc + j0, c5);
    _mm256_storeu_ps(C + (size_t)(i0+6) * ldc + j0, c6);
    _mm256_storeu_ps(C + (size_t)(i0+7) * ldc + j0, c7);
}
#endif
clock_t start;
void sgemm_avx2(
    int M, int N, int K,
    const float* A, int lda,
    const float* B, int ldb,
    float* C, int ldc)
{
    for (int i = 0; i < M; ++i)
        memset(C + (size_t)i * ldc, 0, (size_t)N * sizeof(float));

    start = clock();
#if defined (__AVX2__)
    const int iBlock = 128; 
    const int jBlock = 256; 
    for (int ii = 0; ii < M; ii += iBlock) {
        int iimax = (ii + iBlock < M) ? ii + iBlock : M;
        int i8max = ii + ((iimax - ii) & ~7);
        for (int jj = 0; jj < N; jj += jBlock) {
            int jjmax = (jj + jBlock < N) ? jj + jBlock : N;
            int j8max = jj + ((jjmax - jj) & ~7);

            for (int i = ii; i < i8max; i += 8) {
                for (int j = jj; j < j8max; j += 8) {
                    sgemm_kernel_8x8(K, A, lda, B, ldb, C, ldc, i, j);
                }
                if (j8max < jjmax)
                    sgemm_scalar_block(M, N, K, A, lda, B, ldb, C, ldc,
                                       i, i+8, j8max, jjmax);
            }

            if (i8max < iimax)
                sgemm_scalar_block(M, N, K, A, lda, B, ldb, C, ldc,
                                   i8max, iimax, jj, jjmax);
        }
    }
#else
    for (int i = 0; i < M; ++i) {
        float* c = C + (size_t)i * ldc;
        const float* a = A + (size_t)i * lda;
        for (int k = 0; k < K; ++k) {
            float aik = a[k];
            const float* brow = B + (size_t)k * ldb;
            for (int j = 0; j < N; ++j)
                c[j] += aik * brow[j];
        }
    }
#endif
}

static void fill_rand(float* x, size_t n, unsigned seed) {
    uint32_t s = seed ? seed : 1u;
    for (size_t i = 0; i < n; ++i) {
        s = 1664525u * s + 1013904223u;
        x[i] = (float)((int)(s >> 9) % 2001 - 1000) / 127.0f; // ~[-7.87,7.87]
    }
}

static double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        if (d < 0) d = -d;
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char* argv[]) {
    int M = atoi(argv[1]), N = atoi(argv[3]), K = atoi(argv[2]);
    int lda = K, ldb = N, ldc = N;

    size_t bytesA = (size_t)M * lda * sizeof(float);
    size_t bytesB = (size_t)K * ldb * sizeof(float);
    size_t bytesC = (size_t)M * ldc * sizeof(float);

    float* A = (float*)ALIGN_ALLOC(bytesA, 64);
    float* B = (float*)ALIGN_ALLOC(bytesB, 64);
    float* C = (float*)ALIGN_ALLOC(bytesC, 64);
    float* Cref = (float*)ALIGN_ALLOC(bytesC, 64);
    if (!A || !B || !C || !Cref) { fprintf(stderr, "alloc failed\n"); return 1; }

    fill_rand(A, (size_t)M * K, 1);
    fill_rand(B, (size_t)K * N, 2);
    memset(C, 0, bytesC);
    memset(Cref, 0, bytesC);
    
    
    sgemm_avx2(M, N, K, A, lda, B, ldb, C, ldc);
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", time_taken);
    printf("calculating max abs error: \n");
    
    // Reference (naive) to check correctness
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float aik = A[(size_t)i * lda + k];
            for (int j = 0; j < N; ++j)
                Cref[(size_t)i * ldc + j] += aik * B[(size_t)k * ldb + j];
        }        
    }    
    double err = max_abs_diff(C, Cref, (size_t)M * N);
    printf("Max abs error: %.6g\n", err);

    ALIGN_FREE(A); ALIGN_FREE(B); ALIGN_FREE(C); ALIGN_FREE(Cref);
    return 0;
}
