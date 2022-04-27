#pragma once

void im2col(const float* src, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW);
void naive_conv(const float* src, const float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW);
void im2col_gemm(const float* src, float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW);
void pack1to8_avx(const float* src, float* dst, int inW, int inH, int inC);
void im2col_pack8_avx(const float* src, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW);
void im2col_sgemm_pack8_avx(const float* src, float* kernel, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW);