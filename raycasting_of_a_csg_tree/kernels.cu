#include "kernels.cuh"

__host__ __device__ void MultiplyVectorByMatrix4(float* vector, float* matrix)
{
	float result[4] = { 0 };
	for (int i = 0; i < 4; i++) {
		result[i] = 0;
		for (int j = 0; j < 4; j++) {
			result[i] += vector[j] * matrix[i * 4 + j];
		}
	}
	for (int i = 0; i < 4; i++) {
		vector[i] = result[i];
	}
}

__host__ __device__ void NormalizeVector4(float* vector)
{
	float length = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2] + vector[3] * vector[3]);
	vector[0] /= length;
	vector[1] /= length;
	vector[2] /= length;
	vector[3] /= length;
}

__host__ __device__ void NormalizeVector3(float* vector)
{
	float length = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
	vector[0] /= length;
	vector[1] /= length;
	vector[2] /= length;
}

__host__ __device__ bool TreeContains(Node* tree, float x, float y, float z, int nodeIndex)
{

	if (tree[nodeIndex].left == -1 && tree[nodeIndex].right == -1)
	{
		return SphereContains(tree[nodeIndex].x, tree[nodeIndex].y, tree[nodeIndex].z, tree[nodeIndex].radius, x, y, z);
	}
	else
	{
		bool left = TreeContains(tree, x, y, z, tree[nodeIndex].left);
		bool right = TreeContains(tree, x, y, z, tree[nodeIndex].right);
		if (tree[nodeIndex].operation == 0)
			return SphereSubstraction(left, right);
		else if (tree[nodeIndex].operation == 1)
			return SphereIntersection(left, right);
		else
			return SphereUnion(left, right);
		//return tree[nodeIndex].functionPtr(left, right);
	}
}
__device__ bool BlockingLightRay(DevSphere* spheres, size_t sphere_count, float* pixelPosition, float* lightRay, Node* dev_tree)
{

	pixelPosition[0] += 0.001 * lightRay[0];
	pixelPosition[1] += 0.001 * lightRay[1];
	pixelPosition[2] += 0.001 * lightRay[2];
	for (int k = 0; k < sphere_count; k++)
	{
		float t1, t2;
		if (!IntersectionPoint(&spheres[k], pixelPosition, lightRay, t1, t2)) continue;

		float intersection1[3];
		for (int i = 0; i < 3; i++)
			intersection1[i] = pixelPosition[i] + (t1 + 0.001) * lightRay[i];

		if (t1 > 0 && TreeContains(dev_tree, intersection1[0], intersection1[1], intersection1[2], 0))
		{
			return true;
		}

		float intersection2[3];
		for (int i = 0; i < 3; i++)
			intersection2[i] = pixelPosition[i] + (t2 - 0.001) * lightRay[i];

		if (t2 > 0 && TreeContains(dev_tree, intersection2[0], intersection2[1], intersection2[2], 0))
		{
			return true;
		}
	}
	return false;
}

__global__ void child()
{
	int i = threadIdx.x;
	//printf("Hello from child\n");
}

__global__ void GoTree(Node* arr, float3 point, size_t sphere_count, bool* result)
{
	__shared__ bool results[128];
	//printf("Hello from GoTree\n");
	int index = threadIdx.x + sphere_count - 1;
	if (index >= 2 * sphere_count - 1)
		return;


	// first is a leaf
	results[index] = SphereContains(arr[index].x, arr[index].y, arr[index].z, arr[index].radius, point.x, point.y, point.z);
	__syncthreads();
	//printf("index %d:  %d\n", index, results[index]);

	int prev = index;
	index = arr[index].parent;

	while (index != -1)
	{

		if (arr[index].right == prev) return;

		if (arr[index].operation == 0)
			results[index] = SphereSubstraction(results[arr[index].right], results[arr[index].left]);
		else if (arr[index].operation == 1)
			results[index] = SphereIntersection(results[arr[index].right], results[arr[index].left]);
		else
			results[index] = SphereUnion(results[arr[index].right], results[arr[index].left]);
		__syncthreads();
		//printf("index %d:  %d\n", index, results[index]);

		if (index == 0)
		{
			/**result = results[index];*/
			return;
		}

		prev = index;
		index = arr[index].parent;
	}

}
__global__ void CalculateInterscetion(unsigned char* dev_texture_data, int width, int height, DevSphere* spheres, size_t sphere_count,
	float* pojection, float* view, float* camera_pos, float* light_pos, Node* dev_tree, float* dev_intersecion_points, float* dev_intersection_result)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height)
		return;

	int iindex = (x + y * width) * sphere_count * 2;
	float min = 10000;
	for (int i = 0; i < sphere_count; i++)
	{
		if (x == 400 && y == 300)
		{
			printf("t1: %f, t2: %f\n", dev_intersecion_points[iindex + 2 * i], dev_intersecion_points[iindex + 2 * i + 1]);
		}
		if (dev_intersecion_points[iindex + 2 * i] < min && dev_intersecion_points[iindex+2*i]>0)
			min = dev_intersecion_points[iindex + 2 * i];
	}
		
	if (x == 400 && y == 300)
	{
		printf("min: %f\n", min);
	}
	dev_intersection_result[x + y * width] = min;
}


__global__ void RayWithSphereIntersectionPoints(unsigned char* dev_texture_data, int width, int height, DevSphere* spheres, size_t sphere_count,
	float* projection, float* view, float* camera_pos, float* light_pos, Node* dev_tree, float* dev_intersecion_points)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float stepX = 2 / (float)width;
	float stepY = 2 / (float)height;

	float ray[3] = { -1 + x * stepX, -1 + y * stepY, 1.0f };
	float target[4] = { ray[0], ray[1], ray[2], 1.0f };

	MultiplyVectorByMatrix4(target, projection);
	for (int i = 0; i < 4; i++)
		target[i] /= target[3];
	NormalizeVector4(target);
	target[3] = 0.0f;

	MultiplyVectorByMatrix4(target, view);

	ray[0] = target[0];
	ray[1] = target[1];
	ray[2] = target[2];

	int index = (x + y * width) * sphere_count * 2;
	for (int k = 0; k < sphere_count; k++)
	{
		float t1=-1, t2=-1;
		IntersectionPoint(&spheres[k], camera_pos, ray, t1, t2);

		dev_intersecion_points[index + 2 * k] = t1;
		dev_intersecion_points[index + 2 * k + 1] = t2;
		if (x == 400 && y == 300)
		{
			printf("t1: %f, t2: %f\n", t1, t2);
		}
	}
}

void UpdateOnGPU(unsigned char* dev_texture_data, int width, int height, DevSphere* devSpheres,
	size_t sphere_count, float* projection, float* view, float* camera_pos, float* light_pos, Node* dev_tree, float* dev_intersecion_points, float* dev_intersection_result)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	

	RayWithSphereIntersectionPoints << <grid, block >> > (dev_texture_data, width, height, devSpheres, sphere_count, projection, view, camera_pos, light_pos, dev_tree, dev_intersecion_points);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("RayWithSphereIntersectionPoints launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();

	CalculateInterscetion << <grid, block >> > (dev_texture_data, width, height, devSpheres, sphere_count, projection, view, camera_pos, light_pos, dev_tree, dev_intersecion_points, dev_intersection_result);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CalculateInterscetion launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();

	ColorPixel << <grid, block >> > (dev_texture_data, width, height, devSpheres, sphere_count, projection, view, camera_pos, light_pos, dev_tree, dev_intersecion_points, dev_intersection_result);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CalculateInterscetion launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();

}

__global__ void ColorPixel(unsigned char* dev_texture_data, int width, int height, DevSphere* spheres, size_t sphere_count,
	float* pojection, float* view, float* camera_pos, float* light_pos, Node* dev_tree, float* dev_intersecion_points, float* dev_intersection_result)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float colorf = dev_intersection_result[x + y * width];
	if (x == 400 && y == 300)
	{
		printf("colorf: %f\n", colorf);
	}

	unsigned char color = (colorf < 100 & colorf>0) ? 255 : 0;

	int index = 3 * (y * width + x);
	dev_texture_data[index] = color;
	dev_texture_data[index + 1] = color;
	dev_texture_data[index + 2] = color;
}


__host__ __device__ bool IntersectionPoint(DevSphere* sphere, float* rayOrigin, float* rayDirection, float& t1, float& t2)
{
	float a = dot3(rayDirection, rayDirection);
	float rayMinusSphere[3] = { rayOrigin[0] - sphere->position[0], rayOrigin[1] - sphere->position[1], rayOrigin[2] - sphere->position[2] };
	float b = 2 * dot3(rayDirection, rayMinusSphere);
	float c = dot3(rayMinusSphere, rayMinusSphere) - sphere->radius * sphere->radius;

	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0)
	{
		return false;
	}
	t1 = (-b - sqrt(discriminant)) / (2 * a);
	t2 = (-b + sqrt(discriminant)) / (2 * a);
	return true;
}

__host__ __device__ float dot3(float* a, float* b)
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__host__ __device__ bool SphereSubstraction(bool a, bool b)
{
	return a && !b;
}

__host__ __device__ bool SphereIntersection(bool a, bool b)
{
	return a && b;
}

__host__ __device__ bool SphereUnion(bool a, bool b)
{
	return a || b;
}

__host__ __device__ bool SphereContains(float sx, float sy, float sz, float sr, float x, float y, float z)
{
	return (x - sx) * (x - sx) + (y - sy) * (y - sy) + (z - sz) * (z - sz) < sr * sr;
}