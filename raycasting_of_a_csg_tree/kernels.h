#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

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