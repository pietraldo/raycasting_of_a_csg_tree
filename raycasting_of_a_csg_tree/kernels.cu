#include "kernels.cuh"

__host__ __device__ void MultiplyVectorByMatrix4(float* vector,  float* matrix)
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


__global__ void UpdatePixel(unsigned char* dev_texture_data, int width, int height, DevSphere* spheres, size_t sphere_count,
	float* projection, float* view, float* camera_pos, float* light_pos)
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

	float color[3] = { 0.0f, 0.0f, 0.0f };
	float closest = 1000000;
	for (int k = 0; k < sphere_count; k++)
	{
		float t1, t2;
		if (!IntersectionPoint(&spheres[k], camera_pos, ray, t1, t2)) continue;
		
		if (t1 < closest && t1>0)
		{
			closest = t1;

			float pixelPosition[3];
			for (int i = 0; i < 3; i++)
				pixelPosition[i] = camera_pos[i] + t1 * ray[i];

			bool block = false;

			float lightRay[3] = { light_pos[0] - pixelPosition[0], light_pos[1] - pixelPosition[1], light_pos[2] - pixelPosition[2] };
			float lightDistance = sqrt(lightRay[0] * lightRay[0] + lightRay[1] * lightRay[1] + lightRay[2] * lightRay[2]);
			NormalizeVector3(lightRay);

			for (int l = 0; l < sphere_count; l++)
			{
				if (l == k) continue;
				float t5, t6;
				if (IntersectionPoint(&spheres[l], light_pos, lightRay, t5, t6))
				{
					if (t5 > 0 && t5 < lightDistance)
					{
						block = true;
						break;
					}

				}
			}

			float ka = 0.2; // Ambient reflection coefficient
			float kd = 0.5; // Diffuse reflection coefficient
			float ks = 0.4; // Specular reflection coefficient
			float shininess = 10; // Shininess factor
			float ia = 0.6; // Ambient light intensity
			float id = 0.5; // Diffuse light intensity
			float is = 0.5; // Specular light intensity

			float L[3] = { light_pos[0] - pixelPosition[0], light_pos[1] - pixelPosition[1], light_pos[2] - pixelPosition[2] };
			NormalizeVector3(L);
			float N[3] = { pixelPosition[0] - spheres[k].position[0], pixelPosition[1] - spheres[k].position[1], pixelPosition[2] - spheres[k].position[2] };
			NormalizeVector3(N);
			float V[3] = { -ray[0], -ray[1], -ray[2] };
			NormalizeVector3(V);
			float R[3] = { 2.0f * dot3(L, N) * N[0] - L[0], 2.0f * dot3(L, N) * N[1] - L[1], 2.0f * dot3(L, N) * N[2] - L[2] };
			NormalizeVector3(R);

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
			if (block)
				col = ambient;

			if (col < 0)
				col = 0;
			if (col > 1)
				col = 1;


			color[0] = spheres[k].color[0] * col;
			color[1] = spheres[k].color[1] * col;
			color[2] = spheres[k].color[2] * col;
		}

	}

	int index = 3 * (y * width + x);
	dev_texture_data[index] = color[0];
	dev_texture_data[index + 1] = color[1];
	dev_texture_data[index + 2] = color[2];
}

void UpdateOnGPU(unsigned char* dev_texture_data, int width, int height, DevSphere* devSpheres,
	size_t sphere_count, float* projection, float* view, float* camera_pos, float* light_pos)
{
	dim3 block(16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	UpdatePixel << <grid, block >> > (dev_texture_data, width, height, devSpheres, sphere_count, projection, view, camera_pos, light_pos);
	cudaDeviceSynchronize();
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