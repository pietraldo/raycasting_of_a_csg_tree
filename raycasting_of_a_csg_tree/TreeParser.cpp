#include "TreeParser.h"

TreeParser::TreeParser(string fileName)
{
	this->fileName = fileName;
}



bool TreeParser::Parse()
{
	ifstream file(fileName);
	if (!file.is_open())
	{
		cout << "File not found" << endl;
		return false;
	}

	string line;
	while (getline(file, line))
	{
		try
		{
			if (line[0] == 't')
			{
				istringstream iss(line);
				ParseNode entry;

				iss >> entry.indexStr >> entry.left >> entry.right >> entry.operation;
				entry.index = stoi(entry.indexStr.substr(1, entry.indexStr.find(')')));

				parse_nodes.push_back(entry);
			}
			else if (line[0] == 'c')
			{
				istringstream iss(line);
				ParseCube entry;

				iss >> entry.index >> entry.x >> entry.y >> entry.z >> entry.width >> entry.height >> entry.depth >> entry.r >> entry.g >> entry.b;
				int index = stoi(entry.index.substr(1, entry.index.find(')')));

				float x = entry.x;
				float y = entry.y;
				float z = entry.z;
				float width = entry.width;
				float height = entry.height;
				float depth = entry.depth;

				cubes[index] = Cube{
					make_float3(x, y, z),
					make_float3(x + width, y, z),
					make_float3(x + width, y, z + depth),
					make_float3(x, y, z + depth),
					make_float3(x, y + height, z),
					make_float3(x+width, y + height, z),
					make_float3(x+width, y + height, z+depth),
					make_float3(x, y + height, z+depth),
					make_int3(entry.r, entry.g, entry.b) };
			}
			else if (line[0] == 's')
			{
				istringstream iss(line);
				ParseSphere entry;

				iss >> entry.index >> entry.x >> entry.y >> entry.z >> entry.radius >> entry.r >> entry.g >> entry.b;
				int index = stoi(entry.index.substr(1, entry.index.find(')')));
				spheres[index] = Sphere{ entry.radius, make_float3(entry.x,entry.y,entry.z), make_int3(entry.r,entry.g,entry.b) };
			}
		}
		catch (...)
		{
			cout << "Error parsing file" << endl;
			return false;
		}
	}

	return true;
}