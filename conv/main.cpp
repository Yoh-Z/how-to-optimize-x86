#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

#include "gemm_fun.h"
#include "convolution.h"
using namespace std;

#define DEBUG_TEST 1
#define FUNC_USED mul_v11

double get_current_time()
{
#ifdef _WIN32
	LARGE_INTEGER freq;
	LARGE_INTEGER pc;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&pc);

	return pc.QuadPart * 1000.0 / freq.QuadPart;
#else  // _WIN32
	struct timeval tv;
	gettimeofday(&tv, NULL);

	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif // _WIN32
}

int createRandom(float* src, int width, int height, int bottom, int top)
{
	for (int i = 0; i < height * width; ++i)
	{
		float num = (rand() % (top - bottom + 1)) + bottom;
		*src++ = num;
	}

	return 0;
}

void pretty_print(float* src, int width, int height)
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			cout << src[i * height + j] << " ";
		}
		cout << endl;
	}
}

void gemm_benchmark()
{
	srand((unsigned)time(NULL));
	int m = 800, n = 240, k = 120;
	double gflops = 2.0 * m * n * k * 1.0e-09;
	float* a = new float[m * k];
	float* b = new float[k * n];
	float* c = new float[m * n];
	float* d = new float[m * n];
	memset(c, 0, m * n * sizeof(*c));
	memset(d, 0, m * n * sizeof(*d));

	createRandom(a, m, k, 0, 20);
	createRandom(b, k, n, 0, 20);

	double time_b, time_e;
	time_b = get_current_time();
	FUNC_USED(m, n, k, a, b, c);
	time_e = get_current_time();
	double used_time = (double)(time_e - time_b);


	time_b = get_current_time();
	mul_v1(m, n, k, a, b, d);
	time_e = get_current_time();
	double v1_used_time = (double)(time_e - time_b);
	cout << "mul_v1 " << v1_used_time << endl;

	double diff = 0;
	for (int i = 0; i < m * n; i++)
	{
		diff += abs(c[i] - d[i]);
	}
	if (diff > 0.5)
	{
		cout << diff << " ";
		cout << "error!" << endl;
#if DEBUG_TEST
		pretty_print(a, m, k);
		cout << endl;
		pretty_print(b, k, n);
		cout << endl;
		pretty_print(c, m, n);
		cout << endl;
		pretty_print(d, m, n);
#endif
	}
	else
	{
		for (int i = 0; i < 20; ++i)
		{
			memset(c, 0, m * n * sizeof(*c));
			time_b = get_current_time();
			FUNC_USED(m, n, k, a, b, c);
			time_e = get_current_time();
			used_time = min(used_time, time_e - time_b);
		}
	}

	cout << "used time " << used_time << endl;
	cout << "gf " << gflops / used_time * 1e3 << endl;
	cout << "diff " << diff << endl;

	delete[]a;
	delete[]b;
	delete[]c;
	delete[]d;
}

void test_im2col()
{
	int w = 4, h = 4, c = 3, kW = 3, kH = 3;
	float* input_blob = new float[w * h * c];
	for (int i = 0; i < w * h; i++)
	{
		input_blob[i] = i;
		input_blob[w * h + i] = 2 * i;
		input_blob[w * h * 2 + i] = 3 * i;
	}
	pretty_print(input_blob, w, h);
	pretty_print(input_blob + w * h, w, h);
	pretty_print(input_blob + 2 * w * h, w, h);
	cout << endl;

	const int outW = w - kW + 1;
	const int outH = h - kH + 1;
	float* output_blob = new float[outW * outH * kW * kH * c];
	memset(output_blob, 0, sizeof(float) * outW * outH * kW * kH * c);

	im2col(input_blob, output_blob, w, h, c, kW, kH, 1, 1, 0, 0);

	const int stride = outW * outH * kW * kH;
	for (int k = 0; k < c; k++)
	{
		pretty_print(output_blob + k * stride, kW * kH, outW * outH);
		cout << endl;
	}
}

int main() 
{
	test_im2col();

	return 0;
}
