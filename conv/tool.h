#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

double psnr(cv::Mat& I1, cv::Mat& I2);

void mul_v1(int m, int n, int k, float* a, float* b, float* c);
void mul_v2(int m, int n, int k, float* a, float* b, float* c);
void mul_v3(int m, int n, int k, float* a, float* b, float* c);
void mul_v4(int m, int n, int k, float* a, float* b, float* c);
void mul_v5(int m, int n, int k, float* a, float* b, float* c);
void mul_v6(int m, int n, int k, float* a, float* b, float* c);
void mul_v7(int m, int n, int k, float* a, float* b, float* c);

