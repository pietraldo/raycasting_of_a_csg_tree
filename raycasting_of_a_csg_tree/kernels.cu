#include "kernels.cuh"

__global__ void UpdatePixel(unsigned char* dev_texture_data)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// Check if the thread is within bounds
	if (x < 800 && y < 600) {
		int index = (x + y * 800) * 3; // Compute index for 3-channel texture

		int inx = index / 3;


		int r = blockIdx.x * 5;
		int g = blockIdx.y * 5;
		int b = threadIdx.x * 16;

		// Update texture data
		dev_texture_data[index] = r;
		dev_texture_data[index + 1] = g;
		dev_texture_data[index + 2] = b;
	}
}

void UpdateTextureOnGPU(unsigned char* dev_texture_data) {
	dim3 blockSize(16, 16);
	dim3 gridSize((800 + blockSize.x - 1) / blockSize.x,
		(600 + blockSize.y - 1) / blockSize.y);
	std::cout << gridSize.x << " " << gridSize.y << std::endl;
	UpdatePixel<<<gridSize, blockSize >>>(dev_texture_data);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(error) << std::endl;
	}

	cudaDeviceSynchronize();
	cout << "Hello from GPU" << endl;
}