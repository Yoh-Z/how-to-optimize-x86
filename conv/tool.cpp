#include "tool.h"
#include <immintrin.h>
#include <emmintrin.h>
#include <iostream>

void mul_v1(int m, int n, int k, float* a, float* b, float* c)
{
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            for (int q = 0; q < k; ++q) 
            {
                int idxa = i * k + q;
                int idxb = q * n + j;
                int idxc = i * n + j;
                c[idxc] += a[idxa] * b[idxb];
            }
        }
    }
}


void AddtoDot(int k, float* a, float* b, float* c, int cols_b)
{
    for (int i = 0; i < k; i++) 
    {
        *c += *(a ++) * *(b + cols_b * i);
    }
}

void AddtoDot1x1(int m, int n, int k, float* a, float* b, float* c)
{
    for (int i = 0; i < k; i++)
    {
        *c += *(a ++) * *(b + n * i);
    }
}

void mul_v2(int m, int n, int k, float* a, float* b, float* c)
{
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            AddtoDot1x1(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
}

void mul_v3(int m, int n, int k, float* a, float* b, float* c)
{
    for (int i = 0; i < m; i += 4)
    {
        for(int j = 0; j < n; ++j)
        {
            int idx = i;
            AddtoDot(k, &a[idx * k], &b[j], &c[idx * n + j], n);
            idx++;
            AddtoDot(k, &a[idx * k], &b[j], &c[idx * n + j], n);
            idx++;
            AddtoDot(k, &a[idx * k], &b[j], &c[idx * n + j], n);
            idx++;
            AddtoDot(k, &a[idx * k], &b[j], &c[idx * n + j], n);

        }
    }
}


void AddtoDot1x4(int m, int n, int k, float* a, float* b, float* c)
{
    register float c_00_reg, c_01_reg, c_02_reg, c_03_reg, b_0p_reg;
    float* temp_a1, * temp_a2, * temp_a3, * temp_a4;
    temp_a1 = a;
    temp_a2 = a + k;
    temp_a3 = a + k * 2;
    temp_a4 = a + k * 3;
    c_00_reg = c_01_reg = c_02_reg = c_03_reg = b_0p_reg = 0.0f;

    for (int i = 0; i < k; ++i) 
    {
        b_0p_reg = *(b + i * n);
        c_00_reg += *(temp_a1++) * b_0p_reg;
        c_01_reg += *(temp_a2++) * b_0p_reg;
        c_02_reg += *(temp_a3++) * b_0p_reg;
        c_03_reg += *(temp_a4++) * b_0p_reg;
    }

    *c = c_00_reg;
    *(c + n) = c_01_reg;
    *(c + 2 * n) = c_02_reg;
    *(c + 3 * n) = c_03_reg;
}

void mul_v4(int m, int n, int k, float* a, float* b, float* c)
{
    for (int i = 0; i < m; i += 4)
    {
        for (int j = 0; j < n; ++j)
        {
            AddtoDot1x4(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
}

void AddtoDot4x4(int m, int n, int k, float* a, float* b, float* c)
{
    register float
        c_00_reg, c_01_reg, c_02_reg, c_03_reg,
        c_10_reg, c_11_reg, c_12_reg, c_13_reg,
        c_20_reg, c_21_reg, c_22_reg, c_23_reg,
        c_30_reg, c_31_reg, c_32_reg, c_33_reg;

    float *a_0p_reg, *a_1p_reg, *a_2p_reg, *a_3p_reg;

    c_00_reg = 0.0;   c_01_reg = 0.0;   c_02_reg = 0.0;   c_03_reg = 0.0;
    c_10_reg = 0.0;   c_11_reg = 0.0;   c_12_reg = 0.0;   c_13_reg = 0.0;
    c_20_reg = 0.0;   c_21_reg = 0.0;   c_22_reg = 0.0;   c_23_reg = 0.0;
    c_30_reg = 0.0;   c_31_reg = 0.0;   c_32_reg = 0.0;   c_33_reg = 0.0;

    a_0p_reg = a;
    a_1p_reg = a + k;
    a_2p_reg = a + 2 * k;
    a_3p_reg = a + 3 * k;

    for (int i = 0; i < k; i++)
    {
        float b_00_reg = *(b + i * n),
            b_01_reg = *(b + i * n + 1),
            b_02_reg = *(b + i * n + 2),
            b_03_reg = *(b + i * n + 3);
        c_00_reg += *a_0p_reg * b_00_reg;
        c_01_reg += *a_0p_reg * b_01_reg;
        c_02_reg += *a_0p_reg * b_02_reg;
        c_03_reg += *a_0p_reg++ * b_03_reg;

        c_10_reg += *a_1p_reg * b_00_reg;
        c_11_reg += *a_1p_reg * b_01_reg;
        c_12_reg += *a_1p_reg * b_02_reg;
        c_13_reg += *a_1p_reg++ * b_03_reg;

        c_20_reg += *a_2p_reg * b_00_reg;
        c_21_reg += *a_2p_reg * b_01_reg;
        c_22_reg += *a_2p_reg * b_02_reg;
        c_23_reg += *a_2p_reg++ * b_03_reg;

        c_30_reg += *a_3p_reg * b_00_reg;
        c_31_reg += *a_3p_reg * b_01_reg;
        c_32_reg += *a_3p_reg * b_02_reg;
        c_33_reg += *a_3p_reg++ * b_03_reg;
    }

    float* n_c = c;
    *n_c = c_00_reg, * (n_c + 1) = c_01_reg, * (n_c + 2) = c_02_reg, * (n_c + 3) = c_03_reg;

    n_c = n_c + n;
    *n_c = c_10_reg, * (n_c + 1) = c_11_reg, * (n_c + 2) = c_12_reg, * (n_c + 3) = c_13_reg;

    n_c = n_c + n;
    *n_c = c_20_reg, * (n_c + 1) = c_21_reg, * (n_c + 2) = c_22_reg, * (n_c + 3) = c_23_reg;

    n_c = n_c + n;
    *n_c = c_30_reg, * (n_c + 1) = c_31_reg, * (n_c + 2) = c_32_reg, * (n_c + 3) = c_33_reg;
}


void mul_v5(int m, int n, int k, float* a, float* b, float* c)
{
    const int mm = m - m % 4, nn = n - n % 4;
    for (int i = 0; i < mm; i += 4)
    {
        for (int j = 0; j < nn; j += 4)
        {
            AddtoDot4x4(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
        for (int j = nn; j < n; j++)
        {
            AddtoDot1x4(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
    for (int i = mm; i < m; i++)
    {
        //TODO 4x1
        for (int j = 0; j < n; j++)
        {
            AddtoDot1x1(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }

}

typedef union {
    __m128 v;
    float d[4];
}v4f_t;

void AddtoDot4x4SSE(int m, int n, int k, float* a, float* b, float* c)
{
    float* a_0p_pntr, * a_1p_pntr, * a_2p_pntr, * a_3p_pntr;

    a_0p_pntr = a;
    a_1p_pntr = a + k;
    a_2p_pntr = a + 2 * k;
    a_3p_pntr = a + 3 * k;

    v4f_t
        c_p0_sum,
        c_p1_sum,
        c_p2_sum,
        c_p3_sum;
    v4f_t a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, b_reg;

    c_p0_sum.v = _mm_setzero_ps();
    c_p1_sum.v = _mm_setzero_ps();
    c_p2_sum.v = _mm_setzero_ps();
    c_p3_sum.v = _mm_setzero_ps();
    a_0p_reg.v = _mm_setzero_ps();
    a_1p_reg.v = _mm_setzero_ps();
    a_2p_reg.v = _mm_setzero_ps();
    a_3p_reg.v = _mm_setzero_ps();

    for (int p = 0; p < k; ++p) {
        b_reg.v = _mm_load_ps((float*)&*(b + p * n));

        a_0p_reg.v = _mm_set_ps1(*a_0p_pntr++);
        a_1p_reg.v = _mm_set_ps1(*a_1p_pntr++);
        a_2p_reg.v = _mm_set_ps1(*a_2p_pntr++);
        a_3p_reg.v = _mm_set_ps1(*a_3p_pntr++);

        c_p0_sum.v = _mm_add_ps(c_p0_sum.v, _mm_mul_ps(b_reg.v, a_0p_reg.v));
        c_p1_sum.v = _mm_add_ps(c_p1_sum.v, _mm_mul_ps(b_reg.v, a_1p_reg.v));
        c_p2_sum.v = _mm_add_ps(c_p2_sum.v, _mm_mul_ps(b_reg.v, a_2p_reg.v));
        c_p3_sum.v = _mm_add_ps(c_p3_sum.v, _mm_mul_ps(b_reg.v, a_3p_reg.v));
    }

    float* n_c = c;
    *n_c += c_p0_sum.d[0], * (n_c + 1) += c_p0_sum.d[1], * (n_c + 2) += c_p0_sum.d[2], * (n_c + 3) += c_p0_sum.d[3];

    n_c = n_c + n;
    *n_c += c_p1_sum.d[0], * (n_c + 1) += c_p1_sum.d[1], * (n_c + 2) += c_p1_sum.d[2], * (n_c + 3) += c_p1_sum.d[3];

    n_c = n_c + n;
    *n_c += c_p2_sum.d[0], * (n_c + 1) += c_p2_sum.d[1], * (n_c + 2) += c_p2_sum.d[2], * (n_c + 3) += c_p2_sum.d[3];

    n_c = n_c + n;
    *n_c += c_p3_sum.d[0];
    *(n_c + 1) += c_p3_sum.d[1];
    *(n_c + 2) += c_p3_sum.d[2];
    *(n_c + 3) += c_p3_sum.d[3];
}

void mul_v6(int m, int n, int k, float* a, float* b, float* c)
{
    const int mm = m - m % 4, nn = n - n % 4;
    for (int i = 0; i < mm; i += 4)
    {
        for (int j = 0; j < nn; j += 4)
        {
            AddtoDot4x4SSE(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
        for (int j = nn; j < n; j++)
        {
            AddtoDot1x4(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
    for (int i = mm; i < m; i++)
    {
        //TODO 4x1
        for (int j = 0; j < n; j++)
        {
            AddtoDot1x1(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
}

void AddtoDot4x4FMA(int m, int n, int k, float* a, float* b, float* c)
{
    float* a_0p_pntr, * a_1p_pntr, * a_2p_pntr, * a_3p_pntr;

    a_0p_pntr = a;
    a_1p_pntr = a + k;
    a_2p_pntr = a + 2 * k;
    a_3p_pntr = a + 3 * k;

    v4f_t
        c_p0_sum,
        c_p1_sum,
        c_p2_sum,
        c_p3_sum;
    v4f_t a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, b_reg;

    c_p0_sum.v = _mm_setzero_ps();
    c_p1_sum.v = _mm_setzero_ps();
    c_p2_sum.v = _mm_setzero_ps();
    c_p3_sum.v = _mm_setzero_ps();
    a_0p_reg.v = _mm_setzero_ps();
    a_1p_reg.v = _mm_setzero_ps();
    a_2p_reg.v = _mm_setzero_ps();
    a_3p_reg.v = _mm_setzero_ps();

    for (int p = 0; p < k; ++p) {
        b_reg.v = _mm_load_ps((float*)&*(b + p * n));

        a_0p_reg.v = _mm_set_ps1(*a_0p_pntr++);
        a_1p_reg.v = _mm_set_ps1(*a_1p_pntr++);
        a_2p_reg.v = _mm_set_ps1(*a_2p_pntr++);
        a_3p_reg.v = _mm_set_ps1(*a_3p_pntr++);

        c_p0_sum.v = _mm_fmadd_ps(a_0p_reg.v, b_reg.v, c_p0_sum.v);
        c_p1_sum.v = _mm_fmadd_ps(a_1p_reg.v, b_reg.v, c_p1_sum.v);
        c_p2_sum.v = _mm_fmadd_ps(a_2p_reg.v, b_reg.v, c_p2_sum.v);
        c_p3_sum.v = _mm_fmadd_ps(a_3p_reg.v, b_reg.v, c_p3_sum.v);
    }

    float* n_c = c;
    *n_c += c_p0_sum.d[0], * (n_c + 1) += c_p0_sum.d[1], * (n_c + 2) += c_p0_sum.d[2], * (n_c + 3) += c_p0_sum.d[3];

    n_c = n_c + n;
    *n_c += c_p1_sum.d[0], * (n_c + 1) += c_p1_sum.d[1], * (n_c + 2) += c_p1_sum.d[2], * (n_c + 3) += c_p1_sum.d[3];

    n_c = n_c + n;
    *n_c += c_p2_sum.d[0], * (n_c + 1) += c_p2_sum.d[1], * (n_c + 2) += c_p2_sum.d[2], * (n_c + 3) += c_p2_sum.d[3];

    n_c = n_c + n;
    *n_c += c_p3_sum.d[0], *(n_c + 1) += c_p3_sum.d[1], *(n_c + 2) += c_p3_sum.d[2], *(n_c + 3) += c_p3_sum.d[3];
}

void mul_v7(int m, int n, int k, float* a, float* b, float* c)
{
    const int mm = m - m % 4, nn = n - n % 4;
    for (int i = 0; i < mm; i += 4)
    {
        for (int j = 0; j < nn; j += 4)
        {
            AddtoDot4x4FMA(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
        for (int j = nn; j < n; j++)
        {
            AddtoDot1x4(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
    for (int i = mm; i < m; i++)
    {
        //TODO 4x1
        for (int j = 0; j < n; j++)
        {
            AddtoDot1x1(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
}

typedef union {
    __m256 v;
    float d[8];
}v8f_t;



void AddtoDot8x8AVX(int m, int n, int k, float* a, float* b, float* c)
{
    float* a_0p_pntr, * a_1p_pntr, * a_2p_pntr, * a_3p_pntr, * a_4p_pntr, * a_5p_pntr, * a_6p_pntr, * a_7p_pntr;

    a_0p_pntr = a;
    a_1p_pntr = a + k;
    a_2p_pntr = a + 2 * k;
    a_3p_pntr = a + 3 * k;
    a_4p_pntr = a + 4 * k;
    a_5p_pntr = a + 5 * k;
    a_6p_pntr = a + 6 * k;
    a_7p_pntr = a + 7 * k;

    v8f_t
        c_p0_sum,
        c_p1_sum,
        c_p2_sum,
        c_p3_sum,
        c_p4_sum,
        c_p5_sum,
        c_p6_sum,
        c_p7_sum;

    v8f_t a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, a_4p_reg, a_5p_reg, a_6p_reg, a_7p_reg, b_reg;
    c_p0_sum.v = _mm256_setzero_ps();
    c_p1_sum.v = _mm256_setzero_ps();
    c_p2_sum.v = _mm256_setzero_ps();
    c_p3_sum.v = _mm256_setzero_ps();
    c_p4_sum.v = _mm256_setzero_ps();
    c_p5_sum.v = _mm256_setzero_ps();
    c_p6_sum.v = _mm256_setzero_ps();
    c_p7_sum.v = _mm256_setzero_ps();
    a_0p_reg.v = _mm256_setzero_ps();
    a_1p_reg.v = _mm256_setzero_ps();
    a_2p_reg.v = _mm256_setzero_ps();
    a_3p_reg.v = _mm256_setzero_ps();
    a_4p_reg.v = _mm256_setzero_ps();
    a_5p_reg.v = _mm256_setzero_ps();
    a_6p_reg.v = _mm256_setzero_ps();
    a_7p_reg.v = _mm256_setzero_ps();

    for (int i = 0; i < k; i++)
    {
        b_reg.v = _mm256_load_ps((float*)&b[i * n]);

        a_0p_reg.v = _mm256_set1_ps(*a_0p_pntr++);
        a_1p_reg.v = _mm256_set1_ps(*a_1p_pntr++);
        a_2p_reg.v = _mm256_set1_ps(*a_2p_pntr++);
        a_3p_reg.v = _mm256_set1_ps(*a_3p_pntr++);
        a_4p_reg.v = _mm256_set1_ps(*a_4p_pntr++);
        a_5p_reg.v = _mm256_set1_ps(*a_5p_pntr++);
        a_6p_reg.v = _mm256_set1_ps(*a_6p_pntr++);
        a_7p_reg.v = _mm256_set1_ps(*a_7p_pntr++);

        c_p0_sum.v = _mm256_add_ps(c_p0_sum.v, _mm256_mul_ps(b_reg.v, a_0p_reg.v));
        c_p1_sum.v = _mm256_add_ps(c_p1_sum.v, _mm256_mul_ps(b_reg.v, a_1p_reg.v));
        c_p2_sum.v = _mm256_add_ps(c_p2_sum.v, _mm256_mul_ps(b_reg.v, a_2p_reg.v));
        c_p3_sum.v = _mm256_add_ps(c_p3_sum.v, _mm256_mul_ps(b_reg.v, a_3p_reg.v));
        c_p4_sum.v = _mm256_add_ps(c_p4_sum.v, _mm256_mul_ps(b_reg.v, a_4p_reg.v));
        c_p5_sum.v = _mm256_add_ps(c_p5_sum.v, _mm256_mul_ps(b_reg.v, a_5p_reg.v));
        c_p6_sum.v = _mm256_add_ps(c_p6_sum.v, _mm256_mul_ps(b_reg.v, a_6p_reg.v));
        c_p7_sum.v = _mm256_add_ps(c_p7_sum.v, _mm256_mul_ps(b_reg.v, a_7p_reg.v));
    }

    float* n_c = c;
    _mm256_storeu_ps(n_c, c_p0_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p1_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p2_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p3_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p4_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p5_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p6_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p7_sum.v);
}

void mul_v8(int m, int n, int k, float* a, float* b, float* c)
{
    for (int i = 0; i < m; i += 8)
    {
        for (int j = 0; j < n; j += 8)
        {
            AddtoDot8x8AVX(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
}

void AddtoDot4x8AVX(int m, int n, int k, float* a, float* b, float* c)
{
    float* a_0p_pntr, * a_1p_pntr, * a_2p_pntr, * a_3p_pntr;

    a_0p_pntr = a;
    a_1p_pntr = a + k;
    a_2p_pntr = a + 2 * k;
    a_3p_pntr = a + 3 * k;

    v8f_t
        c_p0_sum,
        c_p1_sum,
        c_p2_sum,
        c_p3_sum;

    v8f_t a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, b_reg;
    c_p0_sum.v = _mm256_setzero_ps();
    c_p1_sum.v = _mm256_setzero_ps();
    c_p2_sum.v = _mm256_setzero_ps();
    c_p3_sum.v = _mm256_setzero_ps();

    a_0p_reg.v = _mm256_setzero_ps();
    a_1p_reg.v = _mm256_setzero_ps();
    a_2p_reg.v = _mm256_setzero_ps();
    a_3p_reg.v = _mm256_setzero_ps();


    for (int i = 0; i < k; i++)
    {
        b_reg.v = _mm256_load_ps((float*)&b[i * n]);

        a_0p_reg.v = _mm256_set1_ps(*a_0p_pntr++);
        a_1p_reg.v = _mm256_set1_ps(*a_1p_pntr++);
        a_2p_reg.v = _mm256_set1_ps(*a_2p_pntr++);
        a_3p_reg.v = _mm256_set1_ps(*a_3p_pntr++);


        c_p0_sum.v = _mm256_add_ps(c_p0_sum.v, _mm256_mul_ps(b_reg.v, a_0p_reg.v));
        c_p1_sum.v = _mm256_add_ps(c_p1_sum.v, _mm256_mul_ps(b_reg.v, a_1p_reg.v));
        c_p2_sum.v = _mm256_add_ps(c_p2_sum.v, _mm256_mul_ps(b_reg.v, a_2p_reg.v));
        c_p3_sum.v = _mm256_add_ps(c_p3_sum.v, _mm256_mul_ps(b_reg.v, a_3p_reg.v));

    }

    float* n_c = c;
    _mm256_storeu_ps(n_c, c_p0_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p1_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p2_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p3_sum.v);

}

void mul_v9(int m, int n, int k, float* a, float* b, float* c)
{
    for (int i = 0; i < m; i += 4)
    {
        for (int j = 0; j < n; j += 8)
        {
            AddtoDot4x8AVX(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
}

void AddtoDot12x8AVX(int m, int n, int k, float* a, float* b, float* c)
{
    float* a_0p_pntr, * a_1p_pntr, * a_2p_pntr, * a_3p_pntr, * a_4p_pntr, * a_5p_pntr, * a_6p_pntr, * a_7p_pntr;

    a_0p_pntr = a;
    a_1p_pntr = a + k;
    a_2p_pntr = a + 2 * k;
    a_3p_pntr = a + 3 * k;
    a_4p_pntr = a + 4 * k;
    a_5p_pntr = a + 5 * k;
    a_6p_pntr = a + 6 * k;
    a_7p_pntr = a + 7 * k;

    v8f_t
        c_p0_sum,
        c_p1_sum,
        c_p2_sum,
        c_p3_sum,
        c_p4_sum,
        c_p5_sum,
        c_p6_sum,
        c_p7_sum;

    v8f_t a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg, a_4p_reg, a_5p_reg, a_6p_reg, a_7p_reg, b_reg;
    c_p0_sum.v = _mm256_setzero_ps();
    c_p1_sum.v = _mm256_setzero_ps();
    c_p2_sum.v = _mm256_setzero_ps();
    c_p3_sum.v = _mm256_setzero_ps();
    c_p4_sum.v = _mm256_setzero_ps();
    c_p5_sum.v = _mm256_setzero_ps();
    c_p6_sum.v = _mm256_setzero_ps();
    c_p7_sum.v = _mm256_setzero_ps();
    a_0p_reg.v = _mm256_setzero_ps();
    a_1p_reg.v = _mm256_setzero_ps();
    a_2p_reg.v = _mm256_setzero_ps();
    a_3p_reg.v = _mm256_setzero_ps();
    a_4p_reg.v = _mm256_setzero_ps();
    a_5p_reg.v = _mm256_setzero_ps();
    a_6p_reg.v = _mm256_setzero_ps();
    a_7p_reg.v = _mm256_setzero_ps();

    for (int i = 0; i < k; i++)
    {
        b_reg.v = _mm256_load_ps((float*)&b[i * n]);

        a_0p_reg.v = _mm256_set1_ps(*a_0p_pntr++);
        a_1p_reg.v = _mm256_set1_ps(*a_1p_pntr++);
        a_2p_reg.v = _mm256_set1_ps(*a_2p_pntr++);
        a_3p_reg.v = _mm256_set1_ps(*a_3p_pntr++);
        a_4p_reg.v = _mm256_set1_ps(*a_4p_pntr++);
        a_5p_reg.v = _mm256_set1_ps(*a_5p_pntr++);
        a_6p_reg.v = _mm256_set1_ps(*a_6p_pntr++);
        a_7p_reg.v = _mm256_set1_ps(*a_7p_pntr++);

        c_p0_sum.v = _mm256_add_ps(c_p0_sum.v, _mm256_mul_ps(b_reg.v, a_0p_reg.v));
        c_p1_sum.v = _mm256_add_ps(c_p1_sum.v, _mm256_mul_ps(b_reg.v, a_1p_reg.v));
        c_p2_sum.v = _mm256_add_ps(c_p2_sum.v, _mm256_mul_ps(b_reg.v, a_2p_reg.v));
        c_p3_sum.v = _mm256_add_ps(c_p3_sum.v, _mm256_mul_ps(b_reg.v, a_3p_reg.v));
        c_p4_sum.v = _mm256_add_ps(c_p4_sum.v, _mm256_mul_ps(b_reg.v, a_4p_reg.v));
        c_p5_sum.v = _mm256_add_ps(c_p5_sum.v, _mm256_mul_ps(b_reg.v, a_5p_reg.v));
        c_p6_sum.v = _mm256_add_ps(c_p6_sum.v, _mm256_mul_ps(b_reg.v, a_6p_reg.v));
        c_p7_sum.v = _mm256_add_ps(c_p7_sum.v, _mm256_mul_ps(b_reg.v, a_7p_reg.v));
    }

    float* n_c = c;
    _mm256_storeu_ps(n_c, c_p0_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p1_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p2_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p3_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p4_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p5_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p6_sum.v);
    n_c += n;
    _mm256_storeu_ps(n_c, c_p7_sum.v);
}

void mul_v10(int m, int n, int k, float* a, float* b, float* c)
{
    for (int i = 0; i < m; i += 12)
    {
        for (int j = 0; j < n; j += 8)
        {
            AddtoDot8x8AVX(m, n, k, &a[i * k], &b[j], &c[i * n + j]);
        }
    }
}