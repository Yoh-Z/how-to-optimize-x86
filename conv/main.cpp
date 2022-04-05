#include <iostream>
#include <cmath>
#include <cstdlib>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

#include "tool.h"
using namespace std;

#define DEBUG_TEST 0
#define FUNC_USED mul_v6

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

int main() 
{
	srand((unsigned)time(NULL));
	int m = 2560, n = 1280, k = 1280;
	double gflops = 2.0 * m * n * k * 1.0e-09;
	float* a = new float[m * k];
	float* b = new float[k * n];
	float* c = new float[m * n];
	memset(c, 0, m * n * sizeof(*c));

	createRandom(a, m, k, 0, 20);
	createRandom(b, k, n, 0, 20);

	double time_b, time_e;
	time_b = get_current_time();
	FUNC_USED(m, n, k, a, b, c);
	time_e = get_current_time();
	double used_time = (double)(time_e - time_b);

	float* d = new float[m * n];
	time_b = get_current_time();
	mul_v1(m, n, k, a, b, d);
	time_e = get_current_time();
	double v1_used_time = (double)(time_e - time_b);
	cout << "mul_v1 " << v1_used_time << endl;
#if DEBUG_TEST
	cout << aP << endl;
	cout << bP << endl;
#endif

	double diff = 0;
	for (int i = 0; i < m * n; i++)
	{
		diff += abs(c[i] - d[i]);
	}
	if (diff > 0.5) 
	{
		cout << diff << " ";
		cout << "error!" << endl;
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
	cout << "gf " << gflops / used_time << endl;
	cout << "diff " << diff << endl;

	getchar();
	return 0;
}