import os
import matplotlib.pyplot as plt

file_path = "data.txt"

fp = open(file_path, "r+")

data_v1 = []
data_v5 = []
data_v6 = []
data_v7 = []
data_v8 = []
data_v11 = []

file_data = fp.readlines()
for data in file_data:
    tmp_data = data.split(' ')
    tmp_data[-1] = tmp_data[-1].replace('\n', '')
    data_v1.append(float(tmp_data[1]))
    data_v5.append(float(tmp_data[2]))
    data_v6.append(float(tmp_data[3]))
    data_v7.append(float(tmp_data[4]))
    data_v8.append(float(tmp_data[5]))
    data_v11.append(float(tmp_data[6]))

x = range(0, 801)
plt.plot(x, data_v1, label='GEMM_v1')
plt.plot(x, data_v5, label='GEMM_v5')
plt.plot(x, data_v6, label='GEMM_v6')
plt.plot(x, data_v7, label='GEMM_v7')
plt.plot(x, data_v8, label='GEMM_v8')
plt.plot(x, data_v11, label='GEMM_v11')
plt.xlabel('M=N=K')
plt.ylabel('gflops')
plt.legend()
plt.show()