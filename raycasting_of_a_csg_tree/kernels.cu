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

__host__ __device__ bool IntersectionPointCube(const Cube& cube, const float3& rayOrigin, const float3& rayDirection, float& t1, float& t2)
{
	float3 l = make_float3(cube.vertices[0].x, cube.vertices[0].y, cube.vertices[0].z);
	float3 h = make_float3(cube.vertices[6].x, cube.vertices[6].y, cube.vertices[6].z);
	float3 o = rayOrigin;
	float3 r = rayDirection;

	float t_close;
	float t_far;

	float tx_low = (l.x - o.x) / r.x;
	float tx_high = (h.x - o.x) / r.x;

	float ty_low = (l.y - o.y) / r.y;
	float ty_high = (h.y - o.y) / r.y;

	float tz_low = (l.z - o.z) / r.z;
	float tz_high = (h.z - o.z) / r.z;

	float tx_close = tx_low < tx_high ? tx_low : tx_high;
	float tx_far = tx_low > tx_high ? tx_low : tx_high;

	float ty_close = ty_low < ty_high ? ty_low : ty_high;
	float ty_far = ty_low > ty_high ? ty_low : ty_high;

	float tz_close = tz_low < tz_high ? tz_low : tz_high;
	float tz_far = tz_low > tz_high ? tz_low : tz_high;

	t_close = tx_close > ty_close ? (tx_close > tz_close ? tx_close : tz_close) : (ty_close > tz_close ? ty_close : tz_close);
	t_far = tx_far < ty_far ? (tx_far < tz_far ? tx_far : tz_far) : (ty_far < tz_far ? ty_far : tz_far);

	t1 = t_close;
	t2 = t_far;

	return t_close < t_far;
}

__global__ void CalculateInterscetion(int width, int height, size_t sphere_count, Node* dev_tree, float* dev_intersecion_points,
	float* dev_intersection_result, int* parts, float* camera_pos_ptr, float* projection, float* view,
	Sphere* dev_spheres, Cube* cubes, unsigned char* dev_texture_data, float* light_pos_ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x >= width || y >= height)
		return;






	float t1 = -1, t2 = -1;
	const int sphereCount = 256; // TODO: change to sphere_count
	float sphereIntersections[2 * sphereCount]; // 2 floats for each sphere
	float tempArray[2 * sphereCount]; // 2 floats for each sphere

	float3 camera_pos = make_float3(camera_pos_ptr[0], camera_pos_ptr[1], camera_pos_ptr[2]);
	float3 light_pos = make_float3(light_pos_ptr[0], light_pos_ptr[1], light_pos_ptr[2]);

	float stepX = 2 / (float)width;
	float stepY = 2 / (float)height;

	float3 ray = make_float3(-1 + x * stepX, -1 + y * stepY, 1.0f);
	float4 target = make_float4(ray.x, ray.y, ray.z, 1.0f);

	MultiplyVectorByMatrix4(target, projection);
	target.x /= target.w;
	target.y /= target.w;
	target.z /= target.w;
	target.w /= target.w;

	target = NormalizeVector4(target);
	target.w = 0.0f;

	MultiplyVectorByMatrix4(target, view);

	ray.x = target.x;
	ray.y = target.y;
	ray.z = target.z;




	Cube cube = cubes[0];

	float3 l = make_float3(cube.vertices[0].x, cube.vertices[0].y, cube.vertices[0].z);
	float3 h = make_float3(cube.vertices[6].x, cube.vertices[6].y, cube.vertices[6].z);
	float3 o = camera_pos;
	float3 r = ray;

	float t_close;
	float t_far;

	float tx_low = (l.x - o.x) / r.x;
	float tx_high = (h.x - o.x) / r.x;

	float ty_low = (l.y - o.y) / r.y;
	float ty_high = (h.y - o.y) / r.y;

	float tz_low = (l.z - o.z) / r.z;
	float tz_high = (h.z - o.z) / r.z;

	float tx_close = tx_low < tx_high ? tx_low : tx_high;
	float tx_far = tx_low > tx_high ? tx_low : tx_high;

	float ty_close = ty_low < ty_high ? ty_low : ty_high;
	float ty_far = ty_low > ty_high ? ty_low : ty_high;

	float tz_close = tz_low < tz_high ? tz_low : tz_high;
	float tz_far = tz_low > tz_high ? tz_low : tz_high;

	t_close = tx_close > ty_close ? (tx_close > tz_close ? tx_close : tz_close) : (ty_close > tz_close ? ty_close : tz_close);
	float t_close_color = tx_close > ty_close ? (tx_close > tz_close ? 10 : 5) : (ty_close > tz_close ? 8 : 5);
	t_far = tx_far < ty_far ? (tx_far < tz_far ? tx_far : tz_far) : (ty_far < tz_far ? ty_far : tz_far);

	float t = t_close;

	if (t_close >= t_far)
	{
		int index2 = 3 * (y * width + x);
		dev_texture_data[index2] = 0;
		dev_texture_data[index2 + 1] = 0;
		dev_texture_data[index2 + 2] = 0;
		return;
	}
	dev_intersection_result[x + y * width] = t_close_color;
	
	float3 N;
	if (tx_close == t_close) {
		N = r.x > 0 ? make_float3(-1, 0, 0) : make_float3(1, 0, 0);
	}
	else if (ty_close == t_close) {
		N = r.y > 0 ? make_float3(0, -1, 0) : make_float3(0, 1, 0);
	}
	else {
		N = r.z > 0 ? make_float3(0, 0, -1) : make_float3(0, 0, 1);
	}


	float3 pixelPosition = make_float3(camera_pos.x + t * ray.x, camera_pos.y + t * ray.y, camera_pos.z + t * ray.z);
	
	float3 lightRay = NormalizeVector3(make_float3(light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z));
	float3 L = NormalizeVector3(make_float3(light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z));
	float3 V = NormalizeVector3(make_float3(-ray.x, -ray.y, -ray.z));
	float3 R = NormalizeVector3(make_float3(2.0f * dot3(L, N) * N.x - L.x, 2.0f * dot3(L, N) * N.y - L.y, 2.0f * dot3(L, N) * N.z - L.z));


	float3 color1 = CalculateColor(N, L, V, R, make_int3(255,255,255));


	int index2 = 3 * (y * width + x);
	dev_texture_data[index2] = (int)color1.x;
	dev_texture_data[index2 + 1] = (int)color1.y;
	dev_texture_data[index2 + 2] = (int)color1.z;

	//unsigned int start = clock();
	//int index = (x + y * width) * sphere_count * 2;
	//for (int k = sphere_count - 1; k < 2 * sphere_count - 1; k++)
	//{
	//	float t1 = -1, t2 = -1;

	//	float3 spherePosition = make_float3(dev_tree[k].sphere->position.x, dev_tree[k].sphere->position.y, dev_tree[k].sphere->position.z);
	//	float radius = dev_tree[k].sphere->radius;
	//	IntersectionPoint(spherePosition, radius, camera_pos, ray, t1, t2);

	//	int m = k - sphere_count + 1;
	//	sphereIntersections[2 * m] = t1;
	//	sphereIntersections[2 * m + 1] = t2;
	//}
	//if (x == 400 && y == 300)
	//{
	//	unsigned int end = clock();
	//	printf("CalculateInterscetion1 time: %f\n", (end - start) / 1000.0f);
	//}

	//unsigned int start2 = clock();
	//for (int i = sphere_count - 2; i >= 0; i--)
	//{
	//	int nodeIndex = i;

	//	if (dev_tree[nodeIndex].operation == 0)
	//	{
	//		int p1 = parts[4 * nodeIndex];
	//		int k1 = parts[4 * nodeIndex + 1];
	//		int p2 = parts[4 * nodeIndex + 2];
	//		int k2 = parts[4 * nodeIndex + 3];

	//		int list1Index = p1;
	//		int list2Index = p2;
	//		int addIndex = p1;


	//		float start1 = sphereIntersections[list1Index];
	//		float end1 = sphereIntersections[list1Index + 1];
	//		float start2 = sphereIntersections[list2Index];
	//		float end2 = sphereIntersections[list2Index + 1];
	//		while (list1Index <= k1 && list2Index <= k2)
	//		{
	//			if (sphereIntersections[list1Index] == -1 || sphereIntersections[list2Index] == -1) // one of the lists just ended
	//			{
	//				break;
	//			}

	//			if (start1 < start2)
	//			{
	//				if (end1 < start2) // przedzaily sie nie nakladaja
	//				{
	//					tempArray[addIndex] = start1;
	//					tempArray[addIndex + 1] = end1;
	//					addIndex += 2;

	//					list1Index += 2;
	//					start1 = sphereIntersections[list1Index];
	//					end1 = sphereIntersections[list1Index + 1];
	//				}
	//				else
	//				{
	//					if (end1 < end2) // usuwa cala koncowke przedzialu
	//					{
	//						tempArray[addIndex] = start1;
	//						tempArray[addIndex + 1] = start2;

	//						addIndex += 2;
	//						list1Index += 2;
	//						start1 = sphereIntersections[list1Index];
	//						end1 = sphereIntersections[list1Index + 1];
	//					}
	//					else // wycina przedzial w srodku
	//					{
	//						tempArray[addIndex] = start1;
	//						tempArray[addIndex + 1] = start2;

	//						addIndex += 2;
	//						start1 = end2;

	//						list2Index += 2;
	//						start2 = sphereIntersections[list2Index + 2];
	//						end2 = sphereIntersections[list2Index + 3];
	//					}
	//				}

	//			}
	//			else
	//			{
	//				if (end2 < start1) // brak przeciecia
	//				{
	//					list2Index += 2;
	//					start2 = sphereIntersections[list2Index];
	//					end2 = sphereIntersections[list2Index + 1];
	//				}
	//				else
	//				{
	//					if (end2 > end1) // usuwa caly przedzial
	//					{
	//						list1Index += 2;
	//						start1 = sphereIntersections[list1Index];
	//						end1 = sphereIntersections[list1Index + 1];
	//					}
	//					else // usuwa poczatek przedzialu
	//					{
	//						start1 = end2;

	//						list2Index += 2;
	//						start2 = sphereIntersections[list2Index];
	//						end2 = sphereIntersections[list2Index + 1];
	//					}
	//				}
	//			}
	//		}




	//		if (list2Index > k2 || sphereIntersections[list2Index] == -1)
	//		{
	//			tempArray[addIndex] = start1;
	//			tempArray[addIndex + 1] = end1;
	//			addIndex += 2;
	//			list1Index += 2;
	//			while (list1Index <= k1 && sphereIntersections[list1Index] != -1)
	//			{
	//				tempArray[addIndex] = sphereIntersections[list1Index];
	//				tempArray[addIndex + 1] = sphereIntersections[list1Index + 1];
	//				addIndex += 2;
	//				list1Index += 2;
	//			}
	//		}

	//		for (int i = p1; i <= k1; i++)
	//		{
	//			if (i < addIndex)
	//				sphereIntersections[i] = tempArray[i];
	//			else
	//				sphereIntersections[i] = -1;
	//		}

	//	}

	//	else if (dev_tree[nodeIndex].operation == 1)
	//	{
	//		int p1 = parts[4 * nodeIndex];
	//		int k1 = parts[4 * nodeIndex + 1];
	//		int p2 = parts[4 * nodeIndex + 2];
	//		int k2 = parts[4 * nodeIndex + 3];

	//		int list1Index = p1;
	//		int list2Index = p2;
	//		int addIndex = p1;

	//		while (list1Index < k1 && list2Index < k2)
	//		{
	//			if (sphereIntersections[list1Index] == -1 || sphereIntersections[list2Index] == -1) // one of the lists just ended
	//			{
	//				break;
	//			}

	//			float start1 = sphereIntersections[list1Index];
	//			float end1 = sphereIntersections[list1Index + 1];
	//			float start2 = sphereIntersections[list2Index];
	//			float end2 = sphereIntersections[list2Index + 1];

	//			if (start1 < start2)
	//			{
	//				if (end1 < start2)
	//				{
	//					list1Index += 2;
	//				}
	//				else
	//				{

	//					if (end1 < end2)
	//					{
	//						tempArray[addIndex] = start2;
	//						tempArray[addIndex + 1] = end1;
	//						addIndex += 2;
	//						list1Index += 2;
	//					}
	//					else
	//					{
	//						tempArray[addIndex] = start2;
	//						tempArray[addIndex + 1] = end2;
	//						addIndex += 2;
	//						list2Index += 2;
	//					}
	//				}
	//			}
	//			else
	//			{
	//				if (end2 < start1)
	//				{
	//					list2Index += 2;
	//				}
	//				else
	//				{
	//					if (end2 < end1)
	//					{
	//						tempArray[addIndex] = start1;
	//						tempArray[addIndex + 1] = end2;
	//						addIndex += 2;
	//						list2Index += 2;
	//					}
	//					else
	//					{
	//						tempArray[addIndex] = start1;
	//						tempArray[addIndex + 1] = end1;
	//						addIndex += 2;
	//						list1Index += 2;
	//					}
	//				}
	//			}
	//		}
	//		for (int i = p1; i <= k1; i++)
	//		{
	//			if (i < addIndex)
	//				sphereIntersections[i] = tempArray[i];
	//			else
	//				sphereIntersections[i] = -1;
	//		}
	//	}

	//	else
	//	{
	//		// TODO: make union
	//		//punkty znajduja sie w lewym od indeksu a do b, w prawym od c do d
	//		int p1 = parts[4 * nodeIndex];
	//		int k1 = parts[4 * nodeIndex + 1];
	//		int p2 = parts[4 * nodeIndex + 2];
	//		int k2 = parts[4 * nodeIndex + 3];



	//		// merging two lists into tempArray sorted by start time
	//		int list1Index = p1;
	//		int list2Index = p2;
	//		int tempIndex = p1;
	//		while (list1Index < k1 && list2Index < k2)
	//		{
	//			if (sphereIntersections[list1Index] == -1 || sphereIntersections[list2Index] == -1) // one of the lists just ended
	//			{
	//				break;
	//			}

	//			if (sphereIntersections[list2Index] < sphereIntersections[list1Index])
	//			{
	//				tempArray[tempIndex] = sphereIntersections[list2Index];
	//				tempArray[tempIndex + 1] = sphereIntersections[list2Index + 1];
	//				list2Index += 2;
	//			}
	//			else
	//			{
	//				tempArray[tempIndex] = sphereIntersections[list1Index];
	//				tempArray[tempIndex + 1] = sphereIntersections[list1Index + 1];
	//				list1Index += 2;
	//			}
	//			tempIndex += 2;
	//		}
	//		while (list1Index < k1 && sphereIntersections[list1Index] != -1)
	//		{
	//			tempArray[tempIndex] = sphereIntersections[list1Index];
	//			tempArray[tempIndex + 1] = sphereIntersections[list1Index + 1];
	//			list1Index += 2;
	//			tempIndex += 2;
	//		}
	//		while (list2Index < k2 && sphereIntersections[list2Index] != -1)
	//		{
	//			tempArray[tempIndex] = sphereIntersections[list2Index];
	//			tempArray[tempIndex + 1] = sphereIntersections[list2Index + 1];
	//			list2Index += 2;
	//			tempIndex += 2;
	//		}



	//		// merging tempArray into sphereIntersections
	//		if (tempIndex != p1) //if something changed
	//		{
	//			float start = tempArray[p1];
	//			float end = tempArray[p1 + 1];
	//			int addIndex = p1;
	//			for (int i = p1 + 2; i <= tempIndex - 2; i += 2)
	//			{
	//				float currentStart = tempArray[i];
	//				float currentEnd = tempArray[i + 1];
	//				if (currentStart > end)
	//				{
	//					sphereIntersections[addIndex] = start;
	//					sphereIntersections[addIndex + 1] = end;
	//					addIndex += 2;
	//					start = currentStart;
	//					end = currentEnd;
	//				}
	//				else
	//				{
	//					if (currentEnd > end)
	//						end = currentEnd;
	//				}
	//			}
	//			sphereIntersections[addIndex] = start;
	//			sphereIntersections[addIndex + 1] = end;
	//			addIndex += 2;


	//			for (int i = addIndex; i <= k2; i++)
	//			{
	//				sphereIntersections[i] = -1;
	//			}
	//		}
	//	}

	//}

	//if (x == 400 && y == 300)
	//{
	//	unsigned int end2 = clock();
	//	printf("CalculateInterscetion2 time: %f\n", (end2 - start2) / 1000.0f);
	//}


	//dev_intersection_result[x + y * width] = sphereIntersections[0] > 0 ? sphereIntersections[0] : 1000;

}


__global__ void RayWithSphereIntersectionPoints(int width, int height, size_t sphere_count,
	float* projection, float* view, float* camera_pos_ptr, Node* dev_tree, float* dev_intersecion_points)
{
	return;
	/*int x = threadIdx.x + blockIdx.x * blockDim.x;
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

	target = NormalizeVector4(target);
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

	}*/
}

void UpdateOnGPU(unsigned char* dev_texture_data, int width, int height,
	size_t sphere_count, float* projection, float* view, float* camera_pos, float* light_pos, Node* dev_tree,
	float* dev_intersecion_points, float* dev_intersection_result, int* dev_parts, Sphere* dev_spheres, Cube* dev_cubes)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);


	auto start2 = std::chrono::high_resolution_clock::now();
	dim3 grid2(width, height);
	CalculateInterscetion << <grid, block >> > (width, height, sphere_count, dev_tree, dev_intersecion_points, dev_intersection_result,
		dev_parts, camera_pos, projection, view, dev_spheres, dev_cubes, dev_texture_data, light_pos);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CalculateInterscetion launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();
	auto end2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed2 = end2 - start2;

	//printf("CalculateInterscetion time: %f\n", elapsed2.count());



	/*auto start3 = std::chrono::high_resolution_clock::now();
	ColorPixel << <grid, block >> > (dev_texture_data, width, height, sphere_count, projection, view, camera_pos, light_pos, dev_tree, dev_intersecion_points, dev_intersection_result, dev_spheres);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CalculateInterscetion launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();
	auto end3 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed3 = end3 - start3;*/

	//printf("ColorPixel time: %f\n", elapsed3.count());

	//printf("%f %f \n", elapsed2.count(), elapsed3.count());

}

__global__ void ColorPixel(unsigned char* dev_texture_data, int width, int height, size_t sphere_count,
	float* projection, float* view, float* camera_pos_ptr, float* light_pos_ptr, Node* dev_tree, float* dev_intersecion_points, float* dev_intersection_result, Sphere* dev_spheres)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x >= width || y >= height)
		return;

	float t = dev_intersection_result[x + y * width];

	if (t == 1000)
	{
		int index2 = 3 * (y * width + x);
		dev_texture_data[index2] = 0;
		dev_texture_data[index2 + 1] = 0;
		dev_texture_data[index2 + 2] = 0;
		return;
	}


	float3 camera_pos = make_float3(camera_pos_ptr[0], camera_pos_ptr[1], camera_pos_ptr[2]);
	float3 light_pos = make_float3(light_pos_ptr[0], light_pos_ptr[1], light_pos_ptr[2]);

	float stepX = 2 / (float)width;
	float stepY = 2 / (float)height;

	float3 ray = make_float3(-1 + x * stepX, -1 + y * stepY, 1.0f);
	float4 target = make_float4(ray.x, ray.y, ray.z, 1.0f);

	MultiplyVectorByMatrix4(target, projection);
	target.x /= target.w;
	target.y /= target.w;
	target.z /= target.w;
	target.w /= target.w;

	target = NormalizeVector4(target);
	target.w = 0.0f;

	MultiplyVectorByMatrix4(target, view);

	ray.x = target.x;
	ray.y = target.y;
	ray.z = target.z;

	{
		//printf("x: %f y: %f z: %f\n", t*ray.x , t*ray.y, t*ray.z);
		unsigned int color = t == 0 ? 0 : (15 - t) / 15 * 255;

		int index2 = 3 * (y * width + x);
		dev_texture_data[index2] = color;
		dev_texture_data[index2 + 1] = color;
		dev_texture_data[index2 + 2] = color;
		return;
	}

	float color[3] = { 0.0f, 0.0f, 0.0f };
	float3 pixelPosition = make_float3(camera_pos.x + t * ray.x, camera_pos.y + t * ray.y, camera_pos.z + t * ray.z);
	float3 N;

	bool intersection = false;
	int index = (x + y * width) * sphere_count * 2;
	int sphereIndex = 0;
	for (int k = sphere_count - 1; k < 2 * sphere_count - 1; k++)
	{
		float t1 = -1, t2 = -1;

		float3 spherePosition = make_float3(dev_tree[k].sphere->position.x, dev_tree[k].sphere->position.y, dev_tree[k].sphere->position.z);
		float radius = dev_tree[k].sphere->radius;
		IntersectionPointSphere(spherePosition, radius, camera_pos, ray, t1, t2);

		if (t1 == t || t2 == t)
		{
			sphereIndex = k - (sphere_count - 1);
		}
		if (t1 == t)
		{
			intersection = true;
			N = NormalizeVector3(make_float3(pixelPosition.x - spherePosition.x, pixelPosition.y - spherePosition.y, pixelPosition.z - spherePosition.z));
			break;
		}
		if (t2 == t)
		{
			intersection = true;
			N = NormalizeVector3(make_float3(-pixelPosition.x + spherePosition.x, -pixelPosition.y + spherePosition.y, -pixelPosition.z + spherePosition.z));
			break;
		}

	}

	if (!intersection) return;

	float3 lightRay = NormalizeVector3(make_float3(light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z));
	float3 L = NormalizeVector3(make_float3(light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z));
	float3 V = NormalizeVector3(make_float3(-ray.x, -ray.y, -ray.z));
	float3 R = NormalizeVector3(make_float3(2.0f * dot3(L, N) * N.x - L.x, 2.0f * dot3(L, N) * N.y - L.y, 2.0f * dot3(L, N) * N.z - L.z));


	float3 color1 = CalculateColor(N, L, V, R, dev_spheres[sphereIndex].color);


	int index2 = 3 * (y * width + x);
	dev_texture_data[index2] = (int)color1.x;
	dev_texture_data[index2 + 1] = (int)color1.y;
	dev_texture_data[index2 + 2] = (int)color1.z;
}

__device__ float3 CalculateColor(const  float3& N, const  float3& L, const  float3& V, const  float3& R, const int3& color)
{
	float ka = 0.2; // Ambient reflection coefficient
	float kd = 0.5; // Diffuse reflection coefficient
	float ks = 0.4; // Specular reflection coefficient
	float shininess = 10; // Shininess factor
	float ia = 0.6; // Ambient light intensity
	float id = 0.5; // Diffuse light intensity
	float is = 0.5; // Specular light intensity

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

	return make_float3(color.x * col, color.y * col, color.z * col);
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