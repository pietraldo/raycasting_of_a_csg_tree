#include "kernels.cuh"

#define NOT_INTERSECTED 1000

#define DEBUG_PIXEL_X 300
#define DEBUG_PIXEL_Y 300


__device__ bool IntersectionPointCube(const Cube& cube, const float3& rayOrigin, const float3& rayDirection, float& t1, float& t2, float3& N, float3& N2)
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

	if (t_close == tx_close)
	{
		if (r.x > 0)
			N = make_float3(-1, 0, 0);
		else
			N = make_float3(1, 0, 0);
	}
	if (t_close == ty_close)
	{
		if (r.y > 0)
			N = make_float3(0, -1, 0);
		else
			N = make_float3(0, 1, 0);
	}
	if (t_close == tz_close)
	{
		if (r.z > 0)
			N = make_float3(0, 0, -1);
		else
			N = make_float3(0, 0, 1);
	}


	if (t_far == tx_far)
	{
		if (r.x > 0)
			N2 = make_float3(-1, 0, 0);
		else
			N2 = make_float3(1, 0, 0);
	}
	if (t_far == ty_far)
	{
		if (r.y > 0)
			N2 = make_float3(0, -1, 0);
		else
			N2 = make_float3(0, 1, 0);
	}
	if (t_far == tz_far)
	{
		if (r.z > 0)
			N2 = make_float3(0, 0, -1);
		else
			N2 = make_float3(0, 0, 1);
	}

	t1 = t_close;
	t2 = t_far;

	return t_close < t_far;
}

__device__ float3 CalculateNormalVectorCylinder(const Cylinder& cylinder, float3 pixelPosition)
{
	float t = dot3(make_float3(pixelPosition.x - cylinder.position.x, pixelPosition.y - cylinder.position.y, pixelPosition.z - cylinder.position.z), cylinder.axis);
	float3 Cp = make_float3(cylinder.position.x + t * cylinder.axis.x, cylinder.position.y + t * cylinder.axis.y, cylinder.position.z + t * cylinder.axis.z);
	float3 r = make_float3(pixelPosition.x - Cp.x, pixelPosition.y - Cp.y, pixelPosition.z - Cp.z);
	return NormalizeVector3(r);
}

__device__ bool IntersectionPointCylinder(const Cylinder& cylinder, const float3& rayOrigin, const float3& rayDirection, float& t1, float& t2, float3& N, float3& N2)
{
	float3 b = make_float3(cylinder.position.x - rayOrigin.x, cylinder.position.y - rayOrigin.y, cylinder.position.z - rayOrigin.z);
	float3 a = NormalizeVector3(cylinder.axis);
	float r = cylinder.radius;
	float h = cylinder.height;
	float3 n = rayDirection;

	float d1 = NOT_INTERSECTED; // in line intersection with cylinder
	float d2 = NOT_INTERSECTED; // out line intersection with cylinder
	float d3 = NOT_INTERSECTED; // first cap with line intersection 
	float d4 = NOT_INTERSECTED; // second cap with line intersection 

	float pierw = dot3(cross(n, a), cross(n, a)) * r * r - dot3(a, a) * dot3(b, cross(n, a)) * dot3(b, cross(n, a));
	if (pierw >= 0)
	{
		d1 = (dot3(cross(n, a), cross(b, a))
			- sqrt(pierw))
			/ dot3(cross(n, a), cross(n, a));
		d2 = (dot3(cross(n, a), cross(b, a))
			+ sqrt(pierw))
			/ dot3(cross(n, a), cross(n, a));

		float t11 = dot3(a, make_float3(n.x * d1 - b.x, n.y * d1 - b.y, n.z * d1 - b.z));
		float t22 = dot3(a, make_float3(n.x * d2 - b.x, n.y * d2 - b.y, n.z * d2 - b.z));

		if (!(t11 >= 0 && t11 <= h)) d1 = NOT_INTERSECTED;
		if (!(t22 >= 0 && t22 <= h)) d2 = NOT_INTERSECTED;
	}

	float3 c1 = b;
	float3 c2 = make_float3(b.x + a.x * h, b.y + a.y * h, b.z + a.z * h);

	d3 = dot3(a, c2) / dot3(a, n);
	d4 = dot3(a, c1) / dot3(a, n);

	if (dot3(make_float3(n.x * d3 - c2.x, n.y * d3 - c2.y, n.z * d3 - c2.z), make_float3(n.x * d3 - c2.x, n.y * d3 - c2.y, n.z * d3 - c2.z)) > r * r || d3 < 0)
		d3 = NOT_INTERSECTED;
	if (dot3(make_float3(n.x * d4 - c1.x, n.y * d4 - c1.y, n.z * d4 - c1.z), make_float3(n.x * d4 - c1.x, n.y * d4 - c1.y, n.z * d4 - c1.z)) > r * r || d4 < 0)
		d4 = NOT_INTERSECTED;

	t1 = NOT_INTERSECTED;
	t2 = NOT_INTERSECTED;
	if (d1 != NOT_INTERSECTED)
	{
		t1 = d1;
		N = CalculateNormalVectorCylinder(cylinder, make_float3(rayOrigin.x + t1 * rayDirection.x, rayOrigin.y + t1 * rayDirection.y, rayOrigin.z + t1 * rayDirection.z));
	}
	if (d3 != NOT_INTERSECTED && d3 < t1)
	{
		t1 = d3;
		N = cylinder.axis;
	}
	if (d4 != NOT_INTERSECTED && d4 < t1)
	{
		t1 = d4;
		N = make_float3(-cylinder.axis.x, -cylinder.axis.y, -cylinder.axis.z);
	}

	// finding smallest t2
	if (d2 != NOT_INTERSECTED)
	{
		t2 = d2;
		N2 = CalculateNormalVectorCylinder(cylinder, make_float3(rayOrigin.x + t2 * rayDirection.x, rayOrigin.y + t2 * rayDirection.y, rayOrigin.z + t2 * rayDirection.z));
		N2 = make_float3(-N2.x, -N2.y, -N2.z);
	}
	if (d3 != NOT_INTERSECTED && d3 < t2 && d3 != t1)
	{
		t2 = d3;
		N2 = make_float3(-cylinder.axis.x, -cylinder.axis.y, -cylinder.axis.z);
	}
	if (d4 != NOT_INTERSECTED && d4 < t2 && d4 != t1)
	{
		t2 = d4;
		N2 = cylinder.axis;
	}

	return true;
}

__device__ void AddIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print)
{
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

__host__ __device__ void AddIntervals2(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print)
{
	bool first = false;
	bool second = false;

	int tempIndex = p1;

	int list1Index = p1;
	int list2Index = p2;

	float start = NOT_INTERSECTED;


	while (true)
	{
		bool list1 = true;
		bool list2 = true;

		if (list1Index > k1 || sphereIntersections[list1Index] == -1)
		{
			list1 = false;
		}
		if (list2Index > k2 || sphereIntersections[list2Index] == -1)
		{
			list2 = false;
		}
		if (!list1 && !list2) break;


		// skiping not detected intersections
		if (list1 && sphereIntersections[list1Index] == NOT_INTERSECTED)
		{
			list1Index += 2;
			continue;
		}
		if (list2 && sphereIntersections[list2Index] == NOT_INTERSECTED)
		{
			list2Index += 2;
			continue;
		}

		if (list1 && list2)
		{
			list1 = sphereIntersections[list1Index] < sphereIntersections[list2Index];
			list2 = !list1;
		}

		if (list1)
		{

			if (!first && start == NOT_INTERSECTED)
			{
				start = sphereIntersections[list1Index];
			}
			else
			{
				if (!second && first)
				{
					tempArray[tempIndex] = start;
					tempArray[tempIndex + 1] = sphereIntersections[list1Index];
					tempIndex += 2;

					start = NOT_INTERSECTED;
				}
			}

			first = !first;
			list1Index++;
		}
		else
		{

			if (!second && start == NOT_INTERSECTED)
			{
				start = sphereIntersections[list2Index];
			}
			else
			{
				if (!first && second)
				{
					tempArray[tempIndex] = start;
					tempArray[tempIndex + 1] = sphereIntersections[list2Index];
					tempIndex += 2;

					start = NOT_INTERSECTED;
				}
			}

			second = !second;
			list2Index++;
		}
	}

	for (int i = p1; i <= k2; i++)
	{
		if (i < tempIndex)
			sphereIntersections[i] = tempArray[i];
		else
			sphereIntersections[i] = -1;
	}
}

__global__ void CalculateInterscetion(int width, int height, int shape_count, Node* dev_tree,
	float* dev_intersection_result, int* parts, float* camera_pos_ptr, float* projection, float* view,
	unsigned char* dev_texture_data, float* light_pos_ptr)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x >= width || y >= height)
		return;


	float t1 = -1, t2 = -1;
	const int sphereCount = 256;
	float sphereIntersections[2 * sphereCount]; // 2 floats for each sphere
	float sphereIntersectionsCopy[2 * sphereCount]; // 2 floats for each sphere
	float3 normalVectors[2 * sphereCount]; // 2 floats for each sphere
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

	for (int k = shape_count - 1; k < 2 * shape_count - 1; k++)
	{
		float t1 = -1, t2 = -1;
		float3 N1, N2;
		if (dev_tree[k].shape == 1)
		{
			float3 spherePosition = make_float3(dev_tree[k].sphere->position.x, dev_tree[k].sphere->position.y, dev_tree[k].sphere->position.z);
			float radius = dev_tree[k].sphere->radius;
			IntersectionPointSphere(spherePosition, radius, camera_pos, ray, t1, t2);

			float3 pixelPosition1 = make_float3(camera_pos.x + t1 * ray.x, camera_pos.y + t1 * ray.y, camera_pos.z + t1 * ray.z);
			float3 pixelPosition2 = make_float3(camera_pos.x + t2 * ray.x, camera_pos.y + t2 * ray.y, camera_pos.z + t2 * ray.z);

			N1 = NormalizeVector3(make_float3(pixelPosition1.x - spherePosition.x, pixelPosition1.y - spherePosition.y, pixelPosition1.z - spherePosition.z));
			N2 = NormalizeVector3(make_float3(-pixelPosition2.x + spherePosition.x, -pixelPosition2.y + spherePosition.y, -pixelPosition2.z + spherePosition.z));
		}
		else if (dev_tree[k].shape == 2)
		{
			Cube* cube = dev_tree[k].cube;
			if (!IntersectionPointCube(*cube, camera_pos, ray, t1, t2, N1, N2))
			{
				t1 = -1;
				t2 = -1;
			}

		}
		else if (dev_tree[k].shape == 3)
		{
			Cylinder* cylinder = dev_tree[k].cylinder;

			if (!IntersectionPointCylinder(*cylinder, camera_pos, ray, t1, t2, N1, N2))
			{
				t1 = -1;
				t2 = -1;
			}
		}

		int m = k - shape_count + 1;


		sphereIntersections[2 * m] = t1;
		sphereIntersections[2 * m + 1] = t2;
		sphereIntersectionsCopy[2 * m] = t1;
		sphereIntersectionsCopy[2 * m + 1] = t2;
		normalVectors[2 * m] = N1;
		normalVectors[2 * m + 1] = N2;
	}




	for (int i = shape_count - 2; i >= 0; i--)
	{
		int nodeIndex = i;

		//punkty znajduja sie w lewym od indeksu a do b, w prawym od c do d
		int p1 = parts[4 * nodeIndex];
		int k1 = parts[4 * nodeIndex + 1];
		int p2 = parts[4 * nodeIndex + 2];
		int k2 = parts[4 * nodeIndex + 3];


		if (dev_tree[nodeIndex].operation == '-')
		{
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
							start2 = sphereIntersections[list2Index];
							end2 = sphereIntersections[list2Index + 1];
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
				if (start1 != end1)
				{
					tempArray[addIndex] = start1;
					tempArray[addIndex + 1] = end1;
					addIndex += 2;
				}
				list1Index += 2;

				while (list1Index <= k1 && sphereIntersections[list1Index] != -1)
				{
					tempArray[addIndex] = sphereIntersections[list1Index];
					tempArray[addIndex + 1] = sphereIntersections[list1Index + 1];
					addIndex += 2;
					list1Index += 2;
				}
			}



			for (int i = p1; i <= k2; i++)
			{
				if (i < addIndex)
					sphereIntersections[i] = tempArray[i];
				else
					sphereIntersections[i] = -1;
			}


		}

		else if (dev_tree[nodeIndex].operation == '*')
		{
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
			for (int i = p2; i <= k2; i++)
			{
				sphereIntersections[i] = -1;
			}
		}

		else
		{
			//DEBUG_PIXEL_X == x && DEBUG_PIXEL_Y == y
			AddIntervals2(sphereIntersections, tempArray, p1, p2, k1, k2, false);
		}

	}


	//dev_intersection_result[x + y * width] = (sphereIntersections[0] > 0 && sphereIntersections[1] != sphereIntersections[0]) ? sphereIntersections[0] : 1000;
	float t = sphereIntersections[0];

	if (t < 0 || sphereIntersections[0] == sphereIntersections[1])
	{
		int index2 = 3 * (y * width + x);
		dev_texture_data[index2] = 0;
		dev_texture_data[index2 + 1] = 0;
		dev_texture_data[index2 + 2] = 0;
		return;
	}

	float3 N = make_float3(0, 0, 1);
	int3 shapeColor = make_int3(0, 0, 0);
	for (int k = shape_count - 1; k < 2 * shape_count - 1; k++)
	{
		int m = k - shape_count + 1;
		if (t != sphereIntersectionsCopy[2 * m] && t != sphereIntersectionsCopy[2 * m + 1]) continue;
		
		if (dev_tree[k].shape == 1)
		{
			shapeColor = dev_tree[k].sphere->color;
		}
		else if (dev_tree[k].shape == 2)
		{
			shapeColor = dev_tree[k].cube->color;
		}
		else if (dev_tree[k].shape == 3)
		{
			shapeColor = dev_tree[k].cylinder->color;
		}
		
		
		if (t == sphereIntersectionsCopy[2 * m])
		{
			N = normalVectors[2 * m];
		}
		else
		{
			N = normalVectors[2 * m + 1];
		}
	}

	float3 pixelPosition = make_float3(camera_pos.x + t * ray.x, camera_pos.y + t * ray.y, camera_pos.z + t * ray.z);
	float3 lightRay = NormalizeVector3(make_float3(light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z));
	float3 L = NormalizeVector3(make_float3(light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z));
	float3 V = NormalizeVector3(make_float3(-ray.x, -ray.y, -ray.z));
	float3 R = NormalizeVector3(make_float3(2.0f * dot3(L, N) * N.x - L.x, 2.0f * dot3(L, N) * N.y - L.y, 2.0f * dot3(L, N) * N.z - L.z));

	float3 color1 = CalculateColor(N, L, V, R, shapeColor);



	int index2 = 3 * (y * width + x);
	dev_texture_data[index2] = (int)color1.x;
	dev_texture_data[index2 + 1] = (int)color1.y;
	dev_texture_data[index2 + 2] = (int)color1.z;
}


void UpdateOnGPU(GPUdata& data, int width, int height)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	dim3 grid2(width, height);
	CalculateInterscetion << <grid, block >> > (width, height, data.ShapeCount, data.dev_tree, data.dev_intersection_result,
		data.dev_parts, data.dev_camera_position, data.dev_projection, data.dev_view, data.dev_texture_data, data.dev_light_postion);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CalculateInterscetion launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();


	/*ColorPixel << <grid, block >> > (data.dev_texture_data, width, height, data.ShapeCount, data.dev_projection,
		data.dev_view, data.dev_camera_position, data.dev_light_postion, data.dev_tree, data.dev_intersection_result);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("ColorPixel launch error: %s\n", cudaGetErrorString(err));
	}
	cudaDeviceSynchronize();*/
}

__global__ void ColorPixel(unsigned char* dev_texture_data, int width, int height, int shape_count,
	float* projection, float* view, float* camera_pos_ptr, float* light_pos_ptr, Node* dev_tree, float* dev_intersection_result)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;


	if (x >= width || y >= height)
		return;

	float t = dev_intersection_result[x + y * width];

	if (x == DEBUG_PIXEL_X && y == DEBUG_PIXEL_Y)
	{
		int index2 = 3 * (y * width + x);
		dev_texture_data[index2] = 255;
		dev_texture_data[index2 + 1] = 0;
		dev_texture_data[index2 + 2] = 255;


		return;
	}

	if (t == NOT_INTERSECTED)
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


	float3 pixelPosition = make_float3(camera_pos.x + t * ray.x, camera_pos.y + t * ray.y, camera_pos.z + t * ray.z);
	float3 N;

	bool hard_shadow = false;
	bool intersection = false;
	int3 shapeColor = make_int3(0, 0, 0);


	for (int k = shape_count - 1; k < 2 * shape_count - 1; k++)
	{
		float t1 = -1, t2 = -1;

		if (dev_tree[k].shape == 1)
		{
			float3 spherePosition = make_float3(dev_tree[k].sphere->position.x, dev_tree[k].sphere->position.y, dev_tree[k].sphere->position.z);
			float radius = dev_tree[k].sphere->radius;
			IntersectionPointSphere(spherePosition, radius, camera_pos, ray, t1, t2);

			if (t1 == t || t2 == t)
			{
				shapeColor = dev_tree[k].sphere->color;
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
		else if (dev_tree[k].shape == 2)
		{
			Cube* cube = dev_tree[k].cube;
			float3 N2;
			if (!IntersectionPointCube(*cube, camera_pos, ray, t1, t2, N, N2)) continue;

			if (t1 == t || t2 == t)
			{
				shapeColor = cube->color;

			}
			if (t1 == t)
			{
				intersection = true;
				break;
			}
			if (t2 == t)
			{
				intersection = true;
				N = N2;
				break;
			}
		}
		else if (dev_tree[k].shape == 3)
		{
			Cylinder* cylinder = dev_tree[k].cylinder;
			float3 N2;
			if (!IntersectionPointCylinder(*cylinder, camera_pos, ray, t1, t2, N, N2)) continue;

			if (t1 == t || t2 == t)
			{
				shapeColor = cylinder->color;
				shapeColor = make_int3(255, 0, 0);
			}
			if (t1 == t)
			{
				intersection = true;
				break;
			}
			if (t2 == t)
			{
				N = N2;
				intersection = true;
				break;
			}
		}
	}

	if (!intersection) return;


	float3 lightRay = NormalizeVector3(make_float3(light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z));
	float3 L = NormalizeVector3(make_float3(light_pos.x - pixelPosition.x, light_pos.y - pixelPosition.y, light_pos.z - pixelPosition.z));
	float3 V = NormalizeVector3(make_float3(-ray.x, -ray.y, -ray.z));
	float3 R = NormalizeVector3(make_float3(2.0f * dot3(L, N) * N.x - L.x, 2.0f * dot3(L, N) * N.y - L.y, 2.0f * dot3(L, N) * N.z - L.z));

	float3 color1 = CalculateColor(N, L, V, R, shapeColor);



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
	float shininess = 100; // Shininess factor
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

__device__ bool IntersectionPointSphere(
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

__host__ __device__ float dot3(const float3& a, const float3& b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 cross(const float3& a, const float3& b)
{
	float3 result;
	result.x = a.y * b.z - a.z * b.y;
	result.y = a.z * b.x - a.x * b.z;
	result.z = a.x * b.y - a.y * b.x;
	return result;
}

