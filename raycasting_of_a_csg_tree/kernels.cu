#include "kernels.cuh"

__host__ __device__ void MultiplyVectorByMatrix4(float4& vector, const float* matrix)
{
	float4 result = { 0, 0, 0, 0 };
	result.x = vector.x * matrix[0] + vector.y * matrix[1] + vector.z * matrix[2] + vector.w * matrix[3];
	result.y = vector.x * matrix[4] + vector.y * matrix[5] + vector.z * matrix[6] + vector.w * matrix[7];
	result.z = vector.x * matrix[8] + vector.y * matrix[9] + vector.z * matrix[10] + vector.w * matrix[11];
	result.w = vector.x * matrix[12] + vector.y * matrix[13] + vector.z * matrix[14] + vector.w * matrix[15];

	vector = result;
}

__host__ __device__ float4 NormalizeVector4(float4 vector)
{
	float length = sqrt(vector.x * vector.x +
		vector.y * vector.y +
		vector.z * vector.z +
		vector.w * vector.w);

	vector.x /= length;
	vector.y /= length;
	vector.z /= length;
	vector.w /= length;

	return vector;
}


__host__ __device__ float3 NormalizeVector3(float3 vector)
{
	float length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
	vector.x /= length;
	vector.y /= length;
	vector.z /= length;
	return vector;
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




				if (list2Index > k2 || sphereIntersections[list2Index] == -1)
				{
					tempArray[addIndex] = start1;
					tempArray[addIndex + 1] = end1;
					addIndex += 2;
					list1Index += 2;
					while (list1Index <= k1 && sphereIntersections[list1Index] != -1)
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
	float* projection, float* view, float* camera_pos_ptr, Node* dev_tree, float* dev_intersecion_points)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float3 camera_pos = make_float3(camera_pos_ptr[0], camera_pos_ptr[1], camera_pos_ptr[2]);

	float stepX = 2 / (float)width;
	float stepY = 2 / (float)height;

	float3 ray = make_float3(-1 + x * stepX, -1 + y * stepY, 1.0f);
	float4 target = make_float4(ray.x, ray.y, ray.z, 1.0f);

	MultiplyVectorByMatrix4(target, projection);
	target.x /= target.w;
	target.y /= target.w;
	target.z /= target.w;
	target.w /= target.w;

	target=NormalizeVector4(target);
	target.w = 0.0f;

	MultiplyVectorByMatrix4(target, view);

	ray.x = target.x;
	ray.y = target.y;
	ray.z = target.z;

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
	float* projection, float* view, float* camera_pos_ptr, float* light_pos_ptr, Node* dev_tree, float* dev_intersecion_points, float* dev_intersection_result)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x >= width || y >= height)
		return;

	float t = dev_intersection_result[x + y * width];

	float3 camera_pos = make_float3(camera_pos_ptr[0], camera_pos_ptr[1], camera_pos_ptr[2]);
	float3 light_pos = make_float3(light_pos_ptr[0], light_pos_ptr[1], light_pos_ptr[2]);

	float stepX = 2 / (float)width;
	float stepY = 2 / (float)height;

	float3 ray = make_float3( - 1 + x * stepX, -1 + y * stepY, 1.0f);
	float4 target = make_float4( ray.x, ray.y, ray.z, 1.0f );

	MultiplyVectorByMatrix4(target, projection);
	target.x /= target.w;
	target.y /= target.w;
	target.z /= target.w;
	target.w /= target.w;

	target=NormalizeVector4(target);
	target.w = 0.0f;

	MultiplyVectorByMatrix4(target, view);

	ray.x = target.x;
	ray.y = target.y;
	ray.z = target.z;

	float color[3] = { 0.0f, 0.0f, 0.0f };
	int index = (x + y * width) * sphere_count * 2;
	for (int k = sphere_count - 1; k < 2 * sphere_count - 1; k++)
	{
		float t1 = -1, t2 = -1;

		float3 spherePosition = make_float3(dev_tree[k].x, dev_tree[k].y, dev_tree[k].z);
		float radius = dev_tree[k].radius;
		IntersectionPoint(spherePosition, radius, camera_pos, ray, t1, t2);

		float3 pixelPosition = make_float3(camera_pos.x + t * ray.x, camera_pos.y + t * ray.y, camera_pos.z + t * ray.z);
		if (t1 == t)
		{
			float3 lightRay = make_float3( light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z );
			float lightDistance = sqrt(lightRay.x * lightRay.x + lightRay.y * lightRay.y + lightRay.z * lightRay.z);
			lightRay=NormalizeVector3(lightRay);

			
			float ka = 0.2; // Ambient reflection coefficient
			float kd = 0.5; // Diffuse reflection coefficient
			float ks = 0.4; // Specular reflection coefficient
			float shininess = 10; // Shininess factor
			float ia = 0.6; // Ambient light intensity
			float id = 0.5; // Diffuse light intensity
			float is = 0.5; // Specular light intensity

			float3 L = make_float3( light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z );
			L=NormalizeVector3(L);
			float3 N = make_float3( pixelPosition.x - spherePosition.x, pixelPosition.y - spherePosition.y, pixelPosition.z - spherePosition.z );
			N=NormalizeVector3(N);
			float3 V = make_float3( - ray.x, -ray.y, -ray.z );
			V=NormalizeVector3(V);
			float3 R = make_float3( 2.0f * dot3(L, N) * N.x - L.x, 2.0f * dot3(L, N) * N.y - L.y, 2.0f * dot3(L, N) * N.z- L.z );
			R=NormalizeVector3(R);

			// Ambient contribution
			float ambient = ka * ia;

			// Diffuse contribution (only if dot(N, L) > 0)
			float diffuse = kd * id * dot3(N, L);
			if (diffuse < 0.0f) {
				diffuse = 0.0f;
			}


			// Specular contribution (only if dot(R, V) > 0)
			float specular = 0.0f;
			float dotRV = dot3(R, V);
			if (dotRV > 0.0f) {
				specular = ks * is * pow(dotRV, shininess);
			}


			float col = ambient + diffuse + specular;

			if (col < 0)
				col = 0;
			if (col > 1)
				col = 1;


			color[0] = 255 * col;
			color[1] = 255 * col;
			color[2] = 255 * col;
		}
		else
		{

		}

	}


	int index2 = 3 * (y * width + x);
	dev_texture_data[index2] = color[0];
	dev_texture_data[index2 + 1] = color[1];
	dev_texture_data[index2 + 2] = color[2];
}


__host__ __device__ bool IntersectionPoint(
	const float3& spherePosition,
	float radius,
	const float3& rayOrigin,
	const float3& rayDirection,
	float& t1,
	float& t2)
{
	// Calculate coefficients for the quadratic equation
	float a = dot3(rayDirection, rayDirection);
	float3 rayMinusSphere = make_float3(
		rayOrigin.x - spherePosition.x,
		rayOrigin.y - spherePosition.y,
		rayOrigin.z - spherePosition.z
	);
	float b = 2.0f * dot3(rayDirection, rayMinusSphere);
	float c = dot3(rayMinusSphere, rayMinusSphere) - radius * radius;

	// Calculate discriminant
	float discriminant = b * b - 4 * a * c;
	if (discriminant < 0.0f)
	{
		return false; // No intersection
	}

	// Calculate t1 and t2 (solutions to the quadratic equation)
	float sqrtDiscriminant = sqrt(discriminant);
	t1 = (-b - sqrtDiscriminant) / (2.0f * a);
	t2 = (-b + sqrtDiscriminant) / (2.0f * a);

	return true; // Intersection found
}

__host__ __device__ float dot3(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
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