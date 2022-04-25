#include <cstring>
#include <cstdio>

#include "convolution.h"
#include "gemm_fun.h"

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

void pretty_print(float* src, int width, int height);

void im2col_gemm(const float* src, float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW)
{
    const int outW = (inW - kW + 2 * paddingW) / strideW + 1;
    const int outH = (inH - kH + 2 * paddingH) / strideH + 1;
    float* im2col_blob = new float[outW * outH * kW * kH * inC];
    memset(im2col_blob, 0, sizeof(float) * outW * outH * kW * kH * inC);
    im2col(src, im2col_blob, inW, inH, inC, kW, kH, 1, 1, 0, 0);

    float* tmp_blob = new float[outW * outH * kW * kH * inC];
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
        mul_v1(outW * outH, 1, kW * kH, tmp_blob + i * outW * outH * kW * kH, kernel + i * kW * kH, dst + i * outW * outH);
    }
}