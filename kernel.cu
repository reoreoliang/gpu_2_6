#include "cuda_runtime.h"
#include "cuda.h"
#include <iostream>
#include "stdio.h"

// 对向量中每一个元素进行平方

constexpr auto N = 5;

__global__ void gpu_square(float* d_in, float* d_out) {
	int tid = threadIdx.x;
	float temp = d_in[tid];
	d_out[tid] = temp * temp;
}

int main(void)
{
	float h_in[N], h_out[N];
	float* d_in, * d_out;

	cudaMalloc((void**)&d_in, N * sizeof(float));
	cudaMalloc((void**)&d_out, N * sizeof(float));
	// 为向量赋值
	for (int i = 0; i < N; i++)
	{
		h_in[i] = i;
	}

	cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
	gpu_square << <1, N >> > (d_in, d_out);
	cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

	printf("Square of Number on GPU\n");
	for (int i = 0; i < N; i++)
	{
		printf("The square of %f is %f\n", h_in[i], h_out[i]);
	}
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
