#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <Windows.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "tool.h"
using namespace cv;
using namespace std;

#define DEBUG_TEST 0
#define FUNC_USED mul_v7

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

	int time_b, time_e;
	time_b = cv::getTickCount();
	FUNC_USED(m, n, k, a, b, c);
	time_e = cv::getTickCount();
	double used_time = (double)(time_e - time_b) / cv::getTickFrequency();

	cv::Mat aP = cv::Mat(m, k, CV_32FC1, a);
	cv::Mat bP = cv::Mat(k, n, CV_32FC1, b);
	time_b = cv::getTickCount();
	cv::Mat cP = aP * bP;
	time_e = cv::getTickCount();
	double opencv_used_time = (double)(time_e - time_b) / cv::getTickFrequency();
	cout << "opencv " << opencv_used_time << endl;
#if DEBUG_TEST
	cout << aP << endl;
	cout << bP << endl;
#endif

	float diff = 0;
	for (int i = 0; i < m; i++) 
	{
		for (int j = 0; j < n; j++) 
		{
			int idx = i * n + j;
			diff += abs(cP.at<float>(i, j) - c[idx]);
#if DEBUG_TEST
			cout << cP.at<float>(i, j) << " " << c[idx] << endl;
#endif
		}
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
			time_b = cv::getTickCount();
			FUNC_USED(m, n, k, a, b, c);
			time_e = cv::getTickCount();
			used_time = min(used_time, (double)(time_e - time_b) / cv::getTickFrequency());
		}
	}

	cout << "used time " << used_time << endl;
	cout << "gf " << gflops / used_time << endl;
	cout << "diff " << diff << endl;
	return 0;
}