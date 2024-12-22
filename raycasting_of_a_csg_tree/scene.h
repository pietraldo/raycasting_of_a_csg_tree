#pragma once


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include "Constants.h"

using namespace glm;

struct Texture {
	GLuint id;
	int width;
	int height;
	int channels;
	std::vector<unsigned char> data;
};

struct Sphere
{
	vec3 position;
	float radius;
};

void SetPixel(Texture& texture, int x, int y, glm::vec3 color)
{
	int index = (y * texture.width + x) * texture.channels;
	texture.data[index] = color.r;
	texture.data[index + 1] = color.g;
	texture.data[index + 2] = color.b;
}

float remap(float a, float b, float t)
{
	return (t - a) / (b - a);
}
struct Camera
{
	vec3 position;
	vec3 direction;
	vec3 up = vec3(0, 1, 0);
    float yaw = 0;
    float pitch=0;
};

glm::mat3 computeRotationMatrix(const glm::vec3& a, const glm::vec3& b) {
    // Normalize the input vectors
    glm::vec3 a_unit = glm::normalize(a);
    glm::vec3 b_unit = glm::normalize(b);

    // Compute the cross product (axis of rotation)
    glm::vec3 v = glm::cross(a_unit, b_unit);

    // Compute the dot product (cosine of the angle)
    float cosTheta = glm::dot(a_unit, b_unit);

    // Special case: vectors are parallel
    if (glm::length(v) < 1e-6f) {
        if (cosTheta > 0.999f) {
            // Vectors are already aligned; return identity matrix
            return glm::mat3(1.0f);
        }
        else {
            // Vectors are anti-parallel; find an orthogonal vector for the axis
            glm::vec3 orthogonal = glm::abs(a_unit.x) < glm::abs(a_unit.y) ?
                glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
            glm::vec3 axis = glm::normalize(glm::cross(a_unit, orthogonal));
            return glm::rotate(glm::mat4(1.0f), glm::pi<float>(), axis);
        }
    }

    // Compute the skew-symmetric matrix for v
    glm::mat3 v_skew = glm::mat3(
        0, -v.z, v.y,
        v.z, 0, -v.x,
        -v.y, v.x, 0
    );

    // Compute the rotation matrix using Rodrigues' formula
    glm::mat3 rotationMatrix = glm::mat3(1.0f) + v_skew + v_skew * v_skew * ((1.0f - cosTheta) / glm::dot(v, v));

    return rotationMatrix;
}

Camera camera = { vec3(0, 0, 0), vec3(0, 0, 1) };
float angleY = 0;
float angleX = 0;
void UpdateTextureCpu(Texture& texture)
{
    //angle += 10;
    camera.direction = normalize(camera.direction);

    float distance = SCR_WIDTH/2;

    vector<Sphere> spheres;
    spheres.push_back({ vec3(0, 0, 7), 1.0f });
    spheres.push_back({ vec3(0, 0, 2), 0.5f });

    float stepX = 2 / (float)TEXTURE_WIDHT;
    float stepY = 2 / (float)TEXTURE_HEIGHT;
    float aspectRatio = (float)TEXTURE_WIDHT / (float)TEXTURE_HEIGHT;

	mat3 rotationMatrix = computeRotationMatrix(vec3(0, 0, 1), camera.direction);

    for (int i = 0; i < TEXTURE_WIDHT; i++)
    {
        for (int j = 0; j < TEXTURE_HEIGHT; j++)
        {
            vec3 color = vec3(0, 0, 0);
            float closest = 1000000;
            for (int k = 0; k < spheres.size(); k++)
            {
                vec3 ray = vec3(i-TEXTURE_WIDHT/2, j-TEXTURE_HEIGHT/2, distance);
                ray = normalize(ray);

				ray = rotationMatrix * ray;

				float radians = glm::radians(angleY);
				float radiansZ = glm::radians(angleX);
                mat3 rotationY = mat3(vec3(cos(radians), 0, sin(radians)), vec3(0, 1, 0), vec3(-sin(radians), 0, cos(radians)));
				mat3 rotationX = mat3(vec3(1, 0, 0), vec3(0, cos(radiansZ), -sin(radiansZ)), vec3(0, sin(radiansZ), cos(radiansZ)));
                ray = rotationX*rotationY * ray;
				/*mat4 view2 = mat4(vec4(camera.position,0), vec4(0, 1, 0, 0), vec4(camera.position + camera.direction,0), vec4(0,0,0,1));
				mat4 view = glm::lookAt(camera.position, camera.position + camera.direction, camera.up);
				vec4 ray4 = vec4(ray, 0);
				ray4 = view * ray4;
				ray = vec3(ray4);*/

                //glm::lookAt(camera.position, camera.position + camera.direction, camera.up);


                float t = dot(spheres[k].position - camera.position, ray);
                vec3 p = camera.position + t * ray;
                float y = length(spheres[k].position - p);
                if (y < spheres[k].radius)
                {
                    float x = sqrt(spheres[k].radius * spheres[k].radius - y * y);
                    float t1 = t - x;

                    if (t1 < closest && t1>0)
                    {
                        closest = t1;
                        float distanceToSphere = length(spheres[k].position - camera.position);
                        float c = remap(distanceToSphere + spheres[k].radius, distanceToSphere - spheres[k].radius, t1);
						if (k == 0)
							color = vec3(255 * c, 255 * c, 255 * c);
						else
                            color = vec3(c * 100, c * 100, 255 * c);
                    }
                }
            }
            SetPixel(texture, i, j, color);
        }
    }
}
