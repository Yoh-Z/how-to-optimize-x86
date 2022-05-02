#include <cstring>
#include <cstdio>
#include <immintrin.h>
#include <emmintrin.h>

#include "convolution.h"
#include "gemm_fun.h"

void pretty_print(float* src, int width, int height);
double get_current_time();

void naive_conv(const float* src, const float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW)
{
    const int outW = (inW - kW + 2 * paddingW) / strideW + 1;
    const int outH = (inH - kH + 2 * paddingH) / strideH + 1;

    float* dst_ptr = dst;
    for (int ch = 0; ch < inC; ch++)
    {
        for (int i = 0; i < outH; i++)
        {
            for (int j = 0; j < outW; j++)
            {
                const float* src_ptr = src + i * inW + j + ch * inW * inH;
                const float* ker_ptr = kernel;

                for (int ii = 0; ii < kH; ii++)
                {
                    for (int jj = 0; jj < kW; jj++)
                    {
                        *dst_ptr += *(src_ptr + jj) * *ker_ptr;
                        ker_ptr++;

                    }
                    src_ptr += inW;
                }
                dst_ptr++;
            }
        }
    }
}

void conv_pack8(const float* src, const float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW)
{
    const int outW = (inW - kW + 2 * paddingW) / strideW + 1;
    const int outH = (inH - kH + 2 * paddingH) / strideH + 1;

    float* dst_ptr = dst;
    for (int ch = 0; ch < inC / 8; ch++)
    {
        for (int i = 0; i < outH; i++)
        {
            for (int j = 0; j < outW; j++)
            {
                const float* src_ptr = src + i * inW + j + ch * inW * inH * 8;
                const float* ker_ptr = kernel;

                __m256 _p = _mm256_load_ps(dst_ptr);
                for (int ii = 0; ii < kH; ii++)
                {
                    for (int jj = 0; jj < kW; jj++)
                    {
                        *dst_ptr += *(src_ptr + jj) * *ker_ptr;
                        
                        __m256 _k = _mm256_load_ps(ker_ptr);
                        __m256 _v = _mm256_load_ps(src_ptr + jj);
                        _p = _mm256_fmadd_ps(_k, _v, _p);
                        
                        ker_ptr += 8;

                    }
                    src_ptr += inW * 8;
                }
                _mm256_store_ps(dst_ptr, _p);
                dst_ptr += 8;
            }
        }
    }
}


void im2col(const float* src, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW)
{
    const int outW = (inW - kW + 2 * paddingW) / strideW + 1;
    const int outH = (inH - kH + 2 * paddingH) / strideH + 1;

    const int gap = (inW * strideH - outW * strideW);

    for (int ch = 0; ch < inC; ch++)
    {
        const float* cur_src = src + ch * inW * inH;
        int dst_idx = kW * kH * outW * outH * ch;

        for (int i = 0; i < kH; i++)
        {
            for (int j = 0; j < kW; j++)
            {
                const float* sptr = cur_src + i * kW + j;

                for (int x = 0; x < outH; x++)
                {
                    for (int y = 0; y < outW; y++)
                    {
                        int row = x * strideH + i;
                        int col = y * strideW + j;
                        int ori_idx = row * inW + col;
                        dst[dst_idx] = cur_src[ori_idx];
                        dst_idx++;
                    }
                }
            }
        }
    }
}

void im2col_gemm(const float* src, float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW)
{
    const int outW = (inW - kW + 2 * paddingW) / strideW + 1;
    const int outH = (inH - kH + 2 * paddingH) / strideH + 1;

    double t1, t2, t3, t4;

    t1 = get_current_time();
    float* tmp_blob = new float[outW * outH * kW * kH * inC];
    float* im2col_blob = new float[outW * outH * kW * kH * inC];
    memset(im2col_blob, 0, sizeof(float) * outW * outH * kW * kH * inC);
    t2 = get_current_time();

    im2col(src, im2col_blob, inW, inH, inC, kW, kH, 1, 1, 0, 0);

    
    float* tmp_ptr = tmp_blob;

    //transpose
    const int stride = outW * outH;
    for (int ch = 0; ch < inC; ch++)
    {
        const float* im2col_ptr = im2col_blob + ch * outW * outH * kW * kH;
        for (int i = 0; i < outW * outH; i++)
        {
            for (int j = 0; j < kH * kW; j++)
            {
                *tmp_ptr++ = *(im2col_ptr + j * stride);
            }
            im2col_ptr++;
        }
    }

    for (int i = 0; i < inC; i++)
    {
        mul_v11(outW * outH, 1, kW * kH, tmp_blob + i * outW * outH * kW * kH, kernel + i * kW * kH, dst + i * outW * outH);
    }

    t3 = get_current_time();
    delete[]im2col_blob;
    delete[]tmp_blob;
    t4 = get_current_time();

    printf("%lf\n", t2 - t1 + t4 - t3);
}

void pack1to8_avx(const float* src, float* dst, int inW, int inH, int inC)
{
    const int nn = inC >> 3;

    const int stride = inW * inH;

    for (int i = 0; i < nn; i++)
    {
        float* cur_dst = dst + i * stride * 8;

        const float* cur_src0 = src + i * stride;
        const float* cur_src1 = cur_src0 + stride;
        const float* cur_src2 = cur_src1 + stride;
        const float* cur_src3 = cur_src2 + stride;
        const float* cur_src4 = cur_src3 + stride;
        const float* cur_src5 = cur_src4 + stride;
        const float* cur_src6 = cur_src5 + stride;
        const float* cur_src7 = cur_src6 + stride;

        for (int j = 0; j < stride; j++)
        {
            *cur_dst++ = cur_src0[j];
            *cur_dst++ = cur_src1[j];
            *cur_dst++ = cur_src2[j];
            *cur_dst++ = cur_src3[j];
            *cur_dst++ = cur_src4[j];
            *cur_dst++ = cur_src5[j];
            *cur_dst++ = cur_src6[j];
            *cur_dst++ = cur_src7[j];
        }
    }
}

void im2col_pack8_avx(const float* src, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW)
{
    const int outW = (inW - kW + 2 * paddingW) / strideW + 1;
    const int outH = (inH - kH + 2 * paddingH) / strideH + 1;

    for (int ch = 0; ch < inC / 8; ch++)
    {
        const float* cur_src = src + ch * inW * inH * 8;
        int dst_idx = kW * kH * outW * outH * ch * 8;

        for (int i = 0; i < kH; i++)
        {
            for (int j = 0; j < kW; j++)
            {
                for (int x = 0; x < outH; x++)
                {
                    for (int y = 0; y < outW; y++)
                    {
                        int row = x * strideH + i;
                        int col = y * strideW + j;
                        int ori_idx = row * inW + col;
                        __m256 _v = _mm256_load_ps(&cur_src[ori_idx * 8]);
                        _mm256_store_ps(&dst[dst_idx], _v);
                        dst_idx += 8;
                    }
                }
            }
        }
    }

}

typedef union {
    __m256 v;
    float d[8];
}v8f_t;

void AddtoDot1x1FMA_pack8(int m, int n, int k, float* a, float* b, float* c)
{
    /*
    (1 2 3 4 5 6 7 8) (2 3 4 5 6 7 8 9) (3 4 5 6 7 8 9 10) (5 x x x x x x x)  (6 x x x x x x x)  (7 x x x x x x x)  (9 x x x x x x x)  (10 x x x x x x x) (11 x x x x x x x)
                                                                                        x
    
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    
                                                                                        |
                                                                                       \_/
    
    (54 x x x x x x x)
    */

    float* a_0p_pntr = a;
    v8f_t c_reg_sum, a_reg, b_reg;
    c_reg_sum.v = _mm256_setzero_ps();
    a_reg.v = _mm256_setzero_ps();

    for (int i = 0; i < k; i++)
    {
        b_reg.v = _mm256_load_ps((float*)&b[i * n]);

        a_reg.v = _mm256_load_ps(a_0p_pntr);

        c_reg_sum.v = _mm256_fmadd_ps(b_reg.v, a_reg.v, c_reg_sum.v);

        a_0p_pntr += 8;
    }

    _mm256_storeu_ps(c, c_reg_sum.v);
}

void AddtoDot1x8FMA_pack8(int m, int n, int k, float* a, float* b, float* c)
{
    /*
    (1 2 3 4 5 6 7 8) (2 3 4 5 6 7 8 9) (3 4 5 6 7 8 9 10) (5 x x x x x x x)  (6 x x x x x x x)  (7 x x x x x x x)  (9 x x x x x x x)  (10 x x x x x x x) (11 x x x x x x x)
    (2 x x x x x x x) (3 x x x x x x x) (4 x x x x x x x)  (6 x x x x x x x)  (7 x x x x x x x)  (8 x x x x x x x)  (10 x x x x x x x) (11 x x x x x x x) (12 x x x x x x x)
    (5 x x x x x x x) (6 x x x x x x x) (7 x x x x x x x)  (9 x x x x x x x)  (10 x x x x x x x) (11 x x x x x x x) (13 x x x x x x x) (14 x x x x x x x) (15 x x x x x x x)
    (6 x x x x x x x) (7 x x x x x x x) (8 x x x x x x x)  (10 x x x x x x x) (11 x x x x x x x) (12 x x x x x x x) (14 x x x x x x x) (15 x x x x x x x) (16 x x x x x x x)
    (1 2 3 4 5 6 7 8) (2 3 4 5 6 7 8 9) (3 4 5 6 7 8 9 10) (5 x x x x x x x)  (6 x x x x x x x)  (7 x x x x x x x)  (9 x x x x x x x)  (10 x x x x x x x) (11 x x x x x x x)
    (2 x x x x x x x) (3 x x x x x x x) (4 x x x x x x x)  (6 x x x x x x x)  (7 x x x x x x x)  (8 x x x x x x x)  (10 x x x x x x x) (11 x x x x x x x) (12 x x x x x x x)
    (5 x x x x x x x) (6 x x x x x x x) (7 x x x x x x x)  (9 x x x x x x x)  (10 x x x x x x x) (11 x x x x x x x) (13 x x x x x x x) (14 x x x x x x x) (15 x x x x x x x)
    (6 x x x x x x x) (7 x x x x x x x) (8 x x x x x x x)  (10 x x x x x x x) (11 x x x x x x x) (12 x x x x x x x) (14 x x x x x x x) (15 x x x x x x x) (16 x x x x x x x)
                                                                                        x
    
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    (1 x x x x x x x)
    
                                                                                        |
                                                                                       \_/
    
    (54 x x x x x x x) (63 x x x x x x x) (90 x x x x x x x) (99 x x x x x x x) (54 x x x x x x x) (63 x x x x x x x) (90 x x x x x x x) (99 x x x x x x x)
    */
    

    float* a_0p_pntr, * a_1p_pntr, * a_2p_pntr, * a_3p_pntr, * a_4p_pntr, * a_5p_pntr, * a_6p_pntr, * a_7p_pntr;

    a_0p_pntr = a;
    a_1p_pntr = a + k * 8;
    a_2p_pntr = a + 2 * k * 8;
    a_3p_pntr = a + 3 * k * 8;
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

        a_0p_reg.v = _mm256_load_ps(a_0p_pntr);
        a_1p_reg.v = _mm256_load_ps(a_1p_pntr);
        a_2p_reg.v = _mm256_load_ps(a_2p_pntr);
        a_3p_reg.v = _mm256_load_ps(a_3p_pntr);
        a_4p_reg.v = _mm256_load_ps(a_4p_pntr);
        a_5p_reg.v = _mm256_load_ps(a_5p_pntr);
        a_6p_reg.v = _mm256_load_ps(a_6p_pntr);
        a_7p_reg.v = _mm256_load_ps(a_7p_pntr);

        c_p0_sum.v = _mm256_fmadd_ps(b_reg.v, a_0p_reg.v, c_p0_sum.v);
        c_p1_sum.v = _mm256_fmadd_ps(b_reg.v, a_1p_reg.v, c_p1_sum.v);
        c_p2_sum.v = _mm256_fmadd_ps(b_reg.v, a_2p_reg.v, c_p2_sum.v);
        c_p3_sum.v = _mm256_fmadd_ps(b_reg.v, a_3p_reg.v, c_p3_sum.v);
        c_p4_sum.v = _mm256_fmadd_ps(b_reg.v, a_4p_reg.v, c_p4_sum.v);
        c_p5_sum.v = _mm256_fmadd_ps(b_reg.v, a_5p_reg.v, c_p5_sum.v);
        c_p6_sum.v = _mm256_fmadd_ps(b_reg.v, a_6p_reg.v, c_p6_sum.v);
        c_p7_sum.v = _mm256_fmadd_ps(b_reg.v, a_7p_reg.v, c_p7_sum.v);

        a_0p_pntr += 8;
        a_1p_pntr += 8;
        a_2p_pntr += 8;
        a_3p_pntr += 8;
        a_4p_pntr += 8;
        a_5p_pntr += 8;
        a_6p_pntr += 8;
        a_7p_pntr += 8;
    }

    float* n_c = c;
    _mm256_storeu_ps(n_c, c_p0_sum.v);
    n_c += n * 8;
    _mm256_storeu_ps(n_c, c_p1_sum.v);
    n_c += n * 8;
    _mm256_storeu_ps(n_c, c_p2_sum.v);
    n_c += n * 8;
    _mm256_storeu_ps(n_c, c_p3_sum.v);
    n_c += n * 8;
    _mm256_storeu_ps(n_c, c_p4_sum.v);
    n_c += n * 8;
    _mm256_storeu_ps(n_c, c_p5_sum.v);
    n_c += n * 8;
    _mm256_storeu_ps(n_c, c_p6_sum.v);
    n_c += n * 8;
    _mm256_storeu_ps(n_c, c_p7_sum.v);


}

void sgemm_pack8_avx(int m, int n, int k, float* a, float* b, float* c)
{
    int i = 0;
    for (; i + 7 < m; i += 8)
    {
        for (int j = 0; j < n; j += 8)
        {
            AddtoDot1x8FMA_pack8(m, n, k, &a[i * k], &b[j], &c[i * n * 8 + j]);
        }
    }
    for (; i < m; i++)
    {
        for (int j = 0; j < n; j += 8)
        {
            AddtoDot1x1FMA_pack8(m, n, k, &a[i * k], &b[j], &c[i * n * 8 + j]);
        }
    }
}

void im2col_sgemm_pack8_avx(const float* src, float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW)
{
    const int outW = (inW - kW + 2 * paddingW) / strideW + 1;
    const int outH = (inH - kH + 2 * paddingH) / strideH + 1;

    float* kernel_tmp = new float[kW * kH * inC];
    pack1to8_avx(kernel, kernel_tmp, kW, kH, inC);

    float* pack8_tmp = new float[inW * inH * inC];
    pack1to8_avx(src, pack8_tmp, inW, inH, inC);
    float* im2col_pack8_blob = new float[kW * kH * inW * inH * inC];
    float* tmp_blob = new float[kW * kH * inW * inH * inC];

    double t1, t2;
    t1 = get_current_time();

    im2col_pack8_avx(pack8_tmp, im2col_pack8_blob, inW, inH, inC, kW, kH, 1, 1, 0, 0);



    //transpose
    float* tmp_ptr = tmp_blob;
    const int stride = outW * outH * 8;
    for (int ch = 0; ch < inC / 8; ch++)
    {
        const float* im2col_ptr = im2col_pack8_blob + ch * outW * outH * kW * kH * 8;
        for (int i = 0; i < outW * outH; i++)
        {
            for (int j = 0; j < kH * kW; j++)
            {
                __m256 _v = _mm256_load_ps(im2col_ptr + j * stride);
                _mm256_store_ps(tmp_ptr, _v);
                tmp_ptr += 8;
            }
            im2col_ptr += 8;
        }
    }

    //pretty_print(tmp_blob, kW * kH * 8, outW * outH);

    for (int i = 0; i < inC / 8; i++)
    {
        sgemm_pack8_avx(outW * outH, 1, kW * kH, tmp_blob + i * outW * outH * kW * kH * 8, kernel + i * kW * kH * 8, dst + i * outW * outH * 8);
    }
    t2 = get_current_time();
    printf("im2col_sgemm_pack8_avx used time :  %lf", t2 - t1);
    //pretty_print(dst, outW * 8, outH);

    delete[]kernel_tmp;
    delete[]pack8_tmp;
    delete[]im2col_pack8_blob;
    delete[]tmp_blob;
}

void conv_pack8_avx(const float* src, float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW)
{
    const int outW = (inW - kW + 2 * paddingW) / strideW + 1;
    const int outH = (inH - kH + 2 * paddingH) / strideH + 1;

    float* kernel_tmp = new float[kW * kH * inC];
    pack1to8_avx(kernel, kernel_tmp, kW, kH, inC);

    float* pack8_tmp = new float[inW * inH * inC];
    pack1to8_avx(src, pack8_tmp, inW, inH, inC);

    double t1, t2;
    t1 = get_current_time();

    conv_pack8(pack8_tmp, kernel_tmp, dst, inW, inH, inC, kW, kH, strideW, strideH, paddingH, paddingW);

    t2 = get_current_time();
    printf("conv_pack8_avx used time :  %lf\n", t2 - t1);

    //pretty_print(dst, outW * 8, outH);
}

