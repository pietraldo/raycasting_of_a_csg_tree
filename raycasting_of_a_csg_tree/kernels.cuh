#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <chrono>

#include <iostream>
#include "DevStruct.h"	
#include "Tree.h"

//__global__ void UpdatePixel(unsigned char* dev_texture_data);
//void UpdateTextureOnGPU(unsigned char* dev_texture_data);




__global__ void RayWithSphereIntersectionPoints(int width, int height, size_t sphere_count,
	float* projection, float* view, float* camera_pos, Node* dev_tree, float* dev_intersecion_points);


__global__ void CalculateInterscetion(int width, int height, size_t sphere_count, Node* dev_tree, float* dev_intersecion_points, float* dev_intersection_result,
	int* parts, float* camera_pos_ptr, float* projection, float* view, Sphere* dev_spheres, Cube* cubes, unsigned char* dev_texture_data, float* light_pos_ptr);

__global__ void ColorPixel(unsigned char* dev_texture_data, int width, int height, size_t sphere_count,
	float* pojection, float* view, float* camera_pos, float* light_pos, Node* dev_tree, float* dev_intersecion_points, float* dev_intersection_result, Sphere* dev_spheres);


void UpdateOnGPU(unsigned char* dev_texture_data, int width, int height,
	size_t sphere_count, float* pojection, float* view, float* camera_pos, float* light_pos, Node* dev_tree,
	float* dev_intersecion_points, float* dev_intersection_result, int* dev_parts, Sphere* dev_spheres, Cube* dev_cubes);


__device__ float3 CalculateColor(const  float3& N, const  float3& L, const  float3& V, const  float3& R,const int3& color);

__host__ __device__ bool IntersectionPointSphere(const float3& spherePosition,float radius,const float3& rayOrigin,const float3& rayDirection,float& t1, float& t2);

__host__ __device__ bool IntersectionPointCube(const Cube& cube, const float3& rayOrigin,const float3& rayDirection,float& t1, float& t2, float3& N, float3& N2);

__host__ __device__ float dot3(const float3& a, const float3& b);

__host__ __device__ void MultiplyVectorByMatrix4(float4& vector, const float* matrix);

__host__ __device__ float4 NormalizeVector4(float4 vector);

__host__ __device__ float3 NormalizeVector3(float3 vector);

__host__ __device__ bool SphereSubstraction(bool a, bool b);

__host__ __device__ bool SphereIntersection(bool a, bool b);

__host__ __device__ bool SphereUnion(bool a, bool b);

__host__ __device__ bool SphereContains(float sx, float sy, float sz, float sr, float x, float y, float z);



//__global__ void UpdatePixel(unsigned char* dev_texture_data, float width, float height, mat4 projection, mat4 view, Sphere* spheres,
//	size_t sphere_count, vec3 camera_pos, vec3 light_pos);