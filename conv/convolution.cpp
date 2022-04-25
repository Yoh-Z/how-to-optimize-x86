#include "convolution.h"
#include "gemm_fun.h"

void naive_conv(float* src, float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW)
{
    const int outW = (inW - kW + 2 * paddingW) / strideW + 1;
    const int outH = (inH - kH + 2 * paddingH) / strideH + 1;

    float* dst_ptr = dst;

    for (int i = 0; i < outH; i++)
    {
        for (int j = 0; j < outW; j++)
        {
            const float* src_ptr = src + i * inW + j;
            const float* ker_ptr = kernel;

            for (int ii = 0; ii < kH; ii++) 
            {
                for(int jj = 0; jj < kH; jj ++)
                {
                    *dst_ptr += *src_ptr * *ker_ptr;
                    dst_ptr++;
                    src_ptr++;
                    ker_ptr++;
                    
                }
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

