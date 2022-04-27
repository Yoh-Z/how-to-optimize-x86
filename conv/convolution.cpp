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

    for (int ch = 0; ch < inC; ch++)
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
                        printf("\n%lf %d %d %d\n", cur_src[ori_idx * 8], ori_idx * 8, row, col);
                        __m256 _v = _mm256_load_ps(&cur_src[ori_idx * 8]);
                        _mm256_store_ps(&dst[dst_idx], _v);
                        dst_idx += 8;
                    }
                }
            }
        }
    }
}