#pragma once

void im2col(const float* src, float* dst, int inW, int inH, int inC, int kW, int kH, int strideW, int strideH, int paddingH, int paddingW);