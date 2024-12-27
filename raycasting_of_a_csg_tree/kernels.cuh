#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <iostream>
#include "DevStruct.h"	

//__global__ void UpdatePixel(unsigned char* dev_texture_data);
//void UpdateTextureOnGPU(unsigned char* dev_texture_data);

__global__ void UpdatePixel(unsigned char* dev_texture_data, int width, int height, DevSphere* spheres, size_t sphere_count,
	float* pojection, float* view, float* camera_pos, float* light_pos);

void UpdateOnGPU(unsigned char* dev_texture_data, int width, int height, DevSphere* devSpheres,
	size_t sphere_count, float* pojection, float* view, float* camera_pos, float* light_pos);

__host__ __device__ bool IntersectionPoint(DevSphere* sphere, float* rayOrigin, float* rayDirection, float& t1, float& t2);

__host__ __device__ float dot3(float* a, float* b);

__host__ __device__ void MultiplyVectorByMatrix4(float* vector, float* matrix);

__host__ __device__ void NormalizeVector4(float* vector);

__host__ __device__ void NormalizeVector3(float* vector);

//__global__ void UpdatePixel(unsigned char* dev_texture_data, float width, float height, mat4 projection, mat4 view, Sphere* spheres,
//	size_t sphere_count, vec3 camera_pos, vec3 light_pos);