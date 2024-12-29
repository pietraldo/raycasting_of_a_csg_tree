#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#include <iostream>
#include "DevStruct.h"	
#include "Tree.h"

//__global__ void UpdatePixel(unsigned char* dev_texture_data);
//void UpdateTextureOnGPU(unsigned char* dev_texture_data);

__global__ void UpdatePixel(unsigned char* dev_texture_data, int width, int height, DevSphere* spheres, size_t sphere_count,
	float* pojection, float* view, float* camera_pos, float* light_pos, Node* dev_tree);

void UpdateOnGPU(unsigned char* dev_texture_data, int width, int height, DevSphere* devSpheres,
	size_t sphere_count, float* pojection, float* view, float* camera_pos, float* light_pos, Node* dev_tree);

__device__ bool BlockingLightRay(DevSphere* spheres, size_t sphere_count, float* pixelPosition, float* lightRay, Node* dev_tree);

__host__ __device__ bool IntersectionPoint(DevSphere* sphere, float* rayOrigin, float* rayDirection, float& t1, float& t2);

__host__ __device__ float dot3(float* a, float* b);

__host__ __device__ void MultiplyVectorByMatrix4(float* vector, float* matrix);

__host__ __device__ void NormalizeVector4(float* vector);

__host__ __device__ void NormalizeVector3(float* vector);

__host__ __device__ bool SphereSubstraction(bool a, bool b);

__host__ __device__ bool SphereIntersection(bool a, bool b);

__host__ __device__ bool SphereUnion(bool a, bool b);

__host__ __device__ bool SphereContains(float sx, float sy, float sz, float sr, float x, float y, float z);

__host__ __device__ bool TreeContains(Node* tree, float x, float y, float z, int nodeIndex);

//__global__ void UpdatePixel(unsigned char* dev_texture_data, float width, float height, mat4 projection, mat4 view, Sphere* spheres,
//	size_t sphere_count, vec3 camera_pos, vec3 light_pos);