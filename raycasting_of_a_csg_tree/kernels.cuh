#pragma once
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include "Tree.h"
#include "GPUdata.h"


__global__ void CalculateInterscetion(int width, int height, int shape_count, Node* dev_tree, float* dev_intersection_result,
	int* parts, float* camera_pos_ptr, float* projection, float* view,unsigned char* dev_texture_data, float* light_pos_ptr);




void UpdateOnGPU(GPUdata& data, int width, int height);

__device__ float3 CalculateNormalVectorCylinder(const Cylinder& cylinder, float3 pixelPosition);
__device__ float3 CalculateColor(const  float3& N, const  float3& L, const  float3& V, const  float3& R,const int3& color);

__device__ bool IntersectionPointSphere(const float3& spherePosition,float radius,const float3& rayOrigin,const float3& rayDirection,float& t1, float& t2);

__device__ bool IntersectionPointCube(const Cube& cube, const float3& rayOrigin,const float3& rayDirection,float& t1, float& t2, float3& N, float3& N2);

__device__ bool IntersectionPointCylinder(const Cylinder& cylinder, const float3& rayOrigin, const float3& rayDirection, float& t1, float& t2, float3& N, float3& N2);

__host__ __device__ void AddIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print);

__host__ __device__ void AddIntervals2(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print);
__host__ __device__ void SubstractIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print);
__host__ __device__ void CommonPartIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print);

__host__ __device__ float4 NormalizeVector4(float4 vector);

__host__ __device__ float3 NormalizeVector3(float3 vector);

__host__ __device__ float3 cross(const float3& a, const float3& b);

__host__ __device__ float dot3(const float3& a, const float3& b);

__host__ __device__ void MultiplyVectorByMatrix4(float4& vector, const float* matrix);