#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <iostream>
using namespace std;


__global__ void UpdatePixel(unsigned char* dev_texture_data);
void UpdateTextureOnGPU(unsigned char* dev_texture_data);
