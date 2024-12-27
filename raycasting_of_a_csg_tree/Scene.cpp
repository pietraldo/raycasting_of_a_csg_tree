#include "scene.h"

void Scene::SetCamera(Camera camera)
{
	this->camera = camera;
}

void Scene::AddSphere(Sphere sphere)
{
	spheres.push_back(sphere);
}

void Scene::SetLight(Light light)
{
	this->light = light;
}

Scene::Scene()
{
	light = Light();
	camera = Camera();
	texture = Texture(3, TEXTURE_WIDHT, TEXTURE_HEIGHT, 0);
}


void Texture::SetPixel(int x, int y, glm::vec3 color)
{
	int index = (y * width + x) * channels;
	data[index] = color.r;
	data[index + 1] = color.g;
	data[index + 2] = color.b;
}

void Scene::UpdateTextureCpu()
{
	vec3 forward = normalize(camera.direction);
	vec3 right = normalize(cross(forward, camera.up));
	vec3 up = normalize(cross(right, forward));

	float translationX = dot(camera.position, right);
	float translationY = dot(camera.position, up);
	float translationZ = dot(camera.position, forward);

	mat4 viewMatrix = mat4(
		vec4(right.x, up.x, -forward.x, 0),
		vec4(right.y, up.y, -forward.y, 0),
		vec4(right.z, up.z, -forward.z, 0),
		vec4(-translationX, -translationY, translationZ, 1)
	);

	// Matrix that transforms from camera space to screen space
	mat4 view = glm::inverse(glm::lookAt(camera.position, camera.position + camera.direction, camera.up));
	mat4 projection = glm::inverse(glm::perspectiveFov(glm::radians(camera.fov), (float)texture.width, (float)texture.height, 0.1f, 10.0f));

	float stepX = 2 / (float)texture.width;
	float stepY = 2 / (float)texture.height;
	for (int i = 0; i < texture.width; i++)
	{
		for (int j = 0; j < texture.height; j++)
		{
			vec3 ray = vec3(-1 + i * stepX, -1 + j * stepY, 1.0f);
			vec4 target = projection * vec4(ray, 1.0f);
			target = normalize(target / target.w);
			target.w = 0.0f;

			ray = view * target;

			vec3 color = vec3(0, 0, 0);
			float closest = 1000000;
			for (int k = 0; k < spheres.size(); k++)
			{
				float t1, t2;
				if (!spheres[k].IntersectionPoint(camera.position, ray, t1, t2)) continue;

				if (t1 < closest && t1>0)
				{
					closest = t1;

					vec3 pixelPosition = camera.position + t1 * ray;

					bool block = false;
					
					vec3 lightRay = pixelPosition - light.position;
					float lightDistance = length(lightRay);
					lightRay = normalize(lightRay);
					
					for (int l = 0; l < spheres.size(); l++)
					{
						if (l == k) continue;
						float t5, t6;
						if (spheres[l].IntersectionPoint(light.position, lightRay, t5, t6))
						{
							if (t5 >0 && t5<lightDistance)
							{
								block = true;
								break;
							}
							
						}
					} 

					float ka = 0.1; // Ambient reflection coefficient
					float kd = 0.5; // Diffuse reflection coefficient
					float ks = 0.4; // Specular reflection coefficient
					float shininess = 10; // Shininess factor
					float ia = 0.5; // Ambient light intensity
					float id = 0.5; // Diffuse light intensity
					float is = 0.5; // Specular light intensity

					vec3 L = normalize(light.position - pixelPosition); // Light direction
					vec3 N = normalize(pixelPosition - spheres[k].position); // Normal at the point
					vec3 V = normalize(-ray); // View direction
					vec3 R = normalize(2.0f * dot(L, N) * N - L); // Reflection vector

					// Ambient contribution
					float ambient = ka * ia;

					// Diffuse contribution (only if dot(N, L) > 0)
					float diffuse = kd * id * dot(N, L);
					if (diffuse < 0.0f) {
						diffuse = 0.0f;
					}


					// Specular contribution (only if dot(R, V) > 0)
					float specular = 0.0f;
					float dotRV = dot(R, V);
					if (dotRV > 0.0f) {
						specular = ks * is * pow(dotRV, shininess);
					}

					
					float col = ambient + diffuse + specular;
					if (block)
						col = ambient;
					col *= 255;
					col = clamp(col, 0.0f, 255.0f);

					color = vec3(col, col, col);
				}

			}
			texture.SetPixel(i, j, color);
		}
	}
}
