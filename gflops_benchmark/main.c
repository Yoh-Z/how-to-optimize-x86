#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

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

#define AVX_FP32_COMP (0x40000000L )
void cpufp_kernel_x86_avx_fp32();

int main()
{
	double t1, t2;
	t1 = get_current_time();
	cpufp_kernel_x86_avx_fp32();
	t2 = get_current_time();

	printf("%lf",  AVX_FP32_COMP / (t2 - t1) * 1e-6 * 96);
}