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

	/*pixelPosition[0] += 0.001 * lightRay[0];
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
	}*/
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


__global__ void CalculateInterscetion(int width, int height, size_t sphere_count, Node* dev_tree, float* dev_intersecion_points,
	float* dev_intersection_result, int* parts)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	if (x >= width || y >= height)
		return;

	if (threadIdx.x >= sphere_count)
		return;

	const int sphereCount = 128; // TODO: change to sphere_count
	__shared__ float sphereIntersections[2 * sphereCount]; // 2 floats for each sphere
	__shared__ float tempArray[2 * sphereCount]; // 2 floats for each sphere
	__shared__ bool isReady[2 * sphereCount - 1];

	if (threadIdx.x == 0)  // Only one thread initializes shared memory
	{
		for (int i = 0; i < 2 * sphere_count - 1; i++)
		{
			isReady[i] = false;
		}
	}

	__syncthreads();


	


	int sphereIndex = threadIdx.x;
	int nodeIndex = sphereIndex + sphere_count - 1;

	float* dev_sphereIntersections = dev_intersecion_points + (x + y * width) * sphere_count * 2;
	float t1 = dev_sphereIntersections[2 * sphereIndex];
	float t2 = dev_sphereIntersections[2 * sphereIndex + 1];

	sphereIntersections[2 * sphereIndex] = t1;
	sphereIntersections[2 * sphereIndex + 1] = t2;

	isReady[nodeIndex] = true;

	__syncthreads();

	/*if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x==0)
	{
		for (int i = 0; i < 2 * sphere_count; i++)
		printf("%d ", isReady[i]);
	}*/

	/*if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
	{
		printf("nodeIndex: %d\n", nodeIndex);
		for (int k = 0; k < 2 * sphere_count; k++)
			printf("%.2f ", sphereIntersections[k]);
		printf("\n");
	}
*/

	int prev = nodeIndex;
	nodeIndex = dev_tree[nodeIndex].parent;


	

	while (nodeIndex != -1)
	{
		/*if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
		{
			printf("nodeIndex: %d\nprzed: ", nodeIndex);
			for (int k = 0; k < 2 * sphere_count; k++)
				printf("%.2f ", sphereIntersections[k]);
			printf("\n");
		}*/
		if (dev_tree[nodeIndex].right == prev) return;

		bool makeOperation = isReady[dev_tree[nodeIndex].right];
		if (makeOperation)
		{
			//if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
				//printf("operaiotn\n");

			if (dev_tree[nodeIndex].operation == 0)
			{
				int p1 = parts[4 * nodeIndex];
				int k1 = parts[4 * nodeIndex + 1];
				int p2 = parts[4 * nodeIndex + 2];
				int k2 = parts[4 * nodeIndex + 3];

				int list1Index = p1;
				int list2Index = p2;
				int addIndex = p1;


				float start1 = sphereIntersections[list1Index];
				float end1 = sphereIntersections[list1Index + 1];
				float start2 = sphereIntersections[list2Index];
				float end2 = sphereIntersections[list2Index + 1];
				while (list1Index <= k1 && list2Index <= k2)
				{
					if (sphereIntersections[list1Index] == -1 || sphereIntersections[list2Index] == -1) // one of the lists just ended
					{
						break;
					}

					if (start1 < start2)
					{
						if (end1 < start2) // przedzaily sie nie nakladaja
						{
							tempArray[addIndex] = start1;
							tempArray[addIndex + 1] = end1;
							addIndex += 2;

							list1Index += 2;
							start1 = sphereIntersections[list1Index];
							end1 = sphereIntersections[list1Index + 1];
						}
						else
						{
							if (end1 < end2) // usuwa cala koncowke przedzialu
							{
								tempArray[addIndex] = start1;
								tempArray[addIndex + 1] = start2;

								addIndex += 2;
								list1Index += 2;
								start1 = sphereIntersections[list1Index];
								end1 = sphereIntersections[list1Index + 1];
							}
							else // wycina przedzial w srodku
							{
								tempArray[addIndex] = start1;
								tempArray[addIndex + 1] = start2;

								addIndex += 2;
								start1 = end2;

								list2Index += 2;
								start2 = sphereIntersections[list2Index + 2];
								end2 = sphereIntersections[list2Index + 3];
							}
						}

					}
					else
					{
						if (end2 < start1) // brak przeciecia
						{
							list2Index += 2;
							start2 = sphereIntersections[list2Index];
							end2 = sphereIntersections[list2Index + 1];
						}
						else
						{
							if (end2 > end1) // usuwa caly przedzial
							{
								list1Index += 2;
								start1 = sphereIntersections[list1Index];
								end1 = sphereIntersections[list1Index + 1];
							}
							else // usuwa poczatek przedzialu
							{
								start1 = end2;

								list2Index += 2;
								start2 = sphereIntersections[list2Index];
								end2 = sphereIntersections[list2Index + 1];
							}
						}
					}
				}


				

				if (list2Index > k2 || sphereIntersections[list2Index]==-1)
				{
					tempArray[addIndex] = start1;
					tempArray[addIndex + 1] = end1;
					addIndex += 2;
					list1Index += 2;
					while (list1Index <= k1 && sphereIntersections[list1Index]!=- 1)
					{
						tempArray[addIndex] = sphereIntersections[list1Index];
						tempArray[addIndex + 1] = sphereIntersections[list1Index + 1];
						addIndex += 2;
						list1Index += 2;
					}
				}
				
				for (int i = p1; i <= k1; i++)
				{
					if (i < addIndex)
						sphereIntersections[i] = tempArray[i];
					else
						sphereIntersections[i] = -1;
				}

			}

			else if (dev_tree[nodeIndex].operation == 1)
			{
				int p1 = parts[4 * nodeIndex];
				int k1 = parts[4 * nodeIndex + 1];
				int p2 = parts[4 * nodeIndex + 2];
				int k2 = parts[4 * nodeIndex + 3];

				int list1Index = p1;
				int list2Index = p2;
				int addIndex = p1;

				while (list1Index < k1 && list2Index < k2)
				{
					if (sphereIntersections[list1Index] == -1 || sphereIntersections[list2Index] == -1) // one of the lists just ended
					{
						break;
					}

					float start1 = sphereIntersections[list1Index];
					float end1 = sphereIntersections[list1Index + 1];
					float start2 = sphereIntersections[list2Index];
					float end2 = sphereIntersections[list2Index + 1];

					if (start1 < start2)
					{
						if (end1 < start2)
						{
							list1Index += 2;
						}
						else
						{

							if (end1 < end2)
							{
								tempArray[addIndex] = start2;
								tempArray[addIndex + 1] = end1;
								addIndex += 2;
								list1Index += 2;
							}
							else
							{
								tempArray[addIndex] = start2;
								tempArray[addIndex + 1] = end2;
								addIndex += 2;
								list2Index += 2;
							}
						}
					}
					else
					{
						if (end2 < start1)
						{
							list2Index += 2;
						}
						else
						{
							if (end2 < end1)
							{
								tempArray[addIndex] = start1;
								tempArray[addIndex + 1] = end2;
								addIndex += 2;
								list2Index += 2;
							}
							else
							{
								tempArray[addIndex] = start1;
								tempArray[addIndex + 1] = end1;
								addIndex += 2;
								list1Index += 2;
							}
						}
					}
				}
				for (int i = p1; i <= k1; i++)
				{
					if (i < addIndex)
						sphereIntersections[i] = tempArray[i];
					else
						sphereIntersections[i] = -1;
				}
			}

			else
			{
				// TODO: make union
				//punkty znajduja sie w lewym od indeksu a do b, w prawym od c do d
				int p1 = parts[4 * nodeIndex];
				int k1 = parts[4 * nodeIndex + 1];
				int p2 = parts[4 * nodeIndex + 2];
				int k2 = parts[4 * nodeIndex + 3];



				// merging two lists into tempArray sorted by start time
				int list1Index = p1;
				int list2Index = p2;
				int tempIndex = p1;
				while (list1Index < k1 && list2Index < k2)
				{
					if (sphereIntersections[list1Index] == -1 || sphereIntersections[list2Index] == -1) // one of the lists just ended
					{
						break;
					}

					if (sphereIntersections[list2Index] < sphereIntersections[list1Index])
					{
						tempArray[tempIndex] = sphereIntersections[list2Index];
						tempArray[tempIndex + 1] = sphereIntersections[list2Index + 1];
						list2Index += 2;
					}
					else
					{
						tempArray[tempIndex] = sphereIntersections[list1Index];
						tempArray[tempIndex + 1] = sphereIntersections[list1Index + 1];
						list1Index += 2;
					}
					tempIndex += 2;
				}
				while (list1Index < k1 && sphereIntersections[list1Index] != -1)
				{
					tempArray[tempIndex] = sphereIntersections[list1Index];
					tempArray[tempIndex + 1] = sphereIntersections[list1Index + 1];
					list1Index += 2;
					tempIndex += 2;
				}
				while (list2Index < k2 && sphereIntersections[list2Index] != -1)
				{
					tempArray[tempIndex] = sphereIntersections[list2Index];
					tempArray[tempIndex + 1] = sphereIntersections[list2Index + 1];
					list2Index += 2;
					tempIndex += 2;
				}

			

				// merging tempArray into sphereIntersections
				if (tempIndex != p1) //if something changed
				{
					float start = tempArray[p1];
					float end = tempArray[p1 + 1];
					int addIndex = p1;
					for (int i = p1 + 2; i <= tempIndex - 2; i += 2)
					{
						float currentStart = tempArray[i];
						float currentEnd = tempArray[i + 1];
						if (currentStart > end)
						{
							sphereIntersections[addIndex] = start;
							sphereIntersections[addIndex + 1] = end;
							addIndex += 2;
							start = currentStart;
							end = currentEnd;
						}
						else
						{
							if (currentEnd > end)
								end = currentEnd;
						}
					}
					sphereIntersections[addIndex] = start;
					sphereIntersections[addIndex + 1] = end;
					addIndex += 2;


					for (int i = addIndex; i <= k2; i++)
					{
						sphereIntersections[i] = -1;
					}
				}
			}
			isReady[nodeIndex] = true;

			/*if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
			{
				printf("po:    ");
				for (int k = 0; k < 2 * sphere_count; k++)
					printf("%.2f ", sphereIntersections[k]);
				printf("\n\n");
			}*/
		}

		

		__syncthreads();

		


		if (makeOperation)
		{
			prev = nodeIndex;
			nodeIndex = dev_tree[nodeIndex].parent;
		}

		/*if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
				printf("operaiotn\n");*/
	}


	dev_intersection_result[x + y * width] = sphereIntersections[0] > 0 ? sphereIntersections[0] : 1000;

}


__global__ void RayWithSphereIntersectionPoints(int width, int height, size_t sphere_count,
	float* projection, float* view, float* camera_pos, Node* dev_tree, float* dev_intersecion_points)
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
	for (int k = sphere_count - 1; k < 2 * sphere_count - 1; k++)
	{
		float t1 = -1, t2 = -1;

		float3 spherePosition = make_float3(dev_tree[k].x, dev_tree[k].y, dev_tree[k].z);
		float radius = dev_tree[k].radius;
		IntersectionPoint(spherePosition, radius, camera_pos, ray, t1, t2);

		int m = k - sphere_count + 1;
		dev_intersecion_points[index + 2 * m] = t1;
		dev_intersecion_points[index + 2 * m + 1] = t2;

	}
}

void UpdateOnGPU(unsigned char* dev_texture_data, int width, int height,
	size_t sphere_count, float* projection, float* view, float* camera_pos, float* light_pos, Node* dev_tree, float* dev_intersecion_points, float* dev_intersection_result, int* dev_parts)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	//printf("RayWithSphereIntersectionPoints started\n");

	RayWithSphereIntersectionPoints << <grid, block >> > (width, height, sphere_count, projection, view, camera_pos, dev_tree, dev_intersecion_points);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("RayWithSphereIntersectionPoints launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();

	//printf("RayWithSphereIntersectionPoints finished\n");

	dim3 grid2(width, height);
	CalculateInterscetion << <grid2, 512 >> > (width, height, sphere_count, dev_tree, dev_intersecion_points, dev_intersection_result, dev_parts);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CalculateInterscetion launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();

	//printf("CalculateInterscetion finished\n");



	ColorPixel << <grid, block >> > (dev_texture_data, width, height, sphere_count, projection, view, camera_pos, light_pos, dev_tree, dev_intersecion_points, dev_intersection_result);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CalculateInterscetion launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();

}

__global__ void ColorPixel(unsigned char* dev_texture_data, int width, int height, size_t sphere_count,
	float* pojection, float* view, float* camera_pos, float* light_pos, Node* dev_tree, float* dev_intersecion_points, float* dev_intersection_result)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x >= width || y >= height)
		return;

	float colorf = (15 - (dev_intersection_result[x + y * width])) / 15.0f * 255;


	unsigned char color = (colorf < 100 & colorf>0) ? 255 : 0;

	color = (int)colorf;

	//if (x == 400 && y == 300)
	//{
	//	printf("dist: %f\n", dev_intersection_result[x + y * width]);
	//}

	int index = 3 * (y * width + x);

	dev_texture_data[index] = color;
	dev_texture_data[index + 1] = color;
	dev_texture_data[index + 2] = color;

	if (x == 400 && y == 300)
	{
		dev_texture_data[index] = 255;
		dev_texture_data[index + 1] = 0;
		dev_texture_data[index + 2] = 0;

	}
}


__host__ __device__ bool IntersectionPoint(float3 spherePosition, float radius, float* rayOrigin, float* rayDirection, float& t1, float& t2)
{
	float a = dot3(rayDirection, rayDirection);
	float rayMinusSphere[3] = { rayOrigin[0] - spherePosition.x, rayOrigin[1] - spherePosition.y, rayOrigin[2] - spherePosition.z };
	float b = 2 * dot3(rayDirection, rayMinusSphere);
	float c = dot3(rayMinusSphere, rayMinusSphere) - radius * radius;

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