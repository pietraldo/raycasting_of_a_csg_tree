#include "TreeParser.h"

TreeParser::TreeParser(string fileName)
{
	this->fileName = fileName;
}

bool TreeParser::CreateObjects()
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

				iss >> entry.indexStr >> entry.left >> entry.operation >> entry.right;
				entry.index = stoi(entry.indexStr.substr(1, entry.indexStr.find(')')));
				entry.indexStr = entry.indexStr.substr(0, entry.indexStr.find(')'));

				parse_nodes.push_back(entry);
				num_nodes++;
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
					make_float3(x + width, y + height, z),
					make_float3(x + width, y + height, z + depth),
					make_float3(x, y + height, z + depth),
					make_int3(entry.r, entry.g, entry.b) };
				num_cubes++;
			}
			else if (line[0] == 's')
			{
				istringstream iss(line);
				ParseSphere entry;

				iss >> entry.index >> entry.x >> entry.y >> entry.z >> entry.radius >> entry.r >> entry.g >> entry.b;
				int index = stoi(entry.index.substr(1, entry.index.find(')')));
				spheres[index] = Sphere{ entry.radius, make_float3(entry.x,entry.y,entry.z), make_int3(entry.r,entry.g,entry.b) };
				num_spheres++;
			}
			else if (line[0] == 'w')
			{
				istringstream iss(line);
				ParseCylinder entry;

				iss >> entry.index >> entry.x >> entry.y >> entry.z >> entry.radius >> entry.height >> entry.axis_x >> entry.axis_y >> entry.axis_z >> entry.r >> entry.g >> entry.b;
				int index = stoi(entry.index.substr(1, entry.index.find(')')));
				cylinders[index] = Cylinder{ entry.radius, entry.height, make_float3(entry.x,entry.y,entry.z), NormalizeVector3(make_float3(entry.axis_x,entry.axis_y,entry.axis_z)), make_int3(entry.r,entry.g,entry.b) };
				num_cylinders++;
			}
		}
		catch (...)
		{
			cout << "Error parsing file" << endl;
			return false;
		}
	}
	return num_spheres + num_cubes +num_cylinders== num_nodes + 1;
}

bool TreeParser::Parse()
{
	if (!CreateObjects())
	{
		return false;
	}


	// find root
	int root = -1;
	for (int i = 0; i < parse_nodes.size(); i++)
	{
		string index = parse_nodes[i].indexStr;
		bool isRoot = true;
		for (int j = 0; j < parse_nodes.size(); j++)
		{
			if (i == j) continue;
			if (parse_nodes[j].left == index || parse_nodes[j].right == index)
			{
				isRoot = false;
				break;
			}
		}
		if (isRoot)
		{
			if (root != -1)
			{
				printf("root: %d\n", root);
				cout << "Multiple roots found" << endl;
				return false;
			}
			else
			{
				root = i;
				printf("root: %d\n", root);
			}
		}
	}

	vector<string> indexes = vector<string>();

	stack<string> stack;
	stack.push(parse_nodes[root].indexStr);

	int i = 0;
	while (!stack.empty())
	{
		string index = stack.top();
		stack.pop();
		
		if (index[0] != 't')
		{
			leavesIndexes.push_back(index);
			continue;
		}
		indexes.push_back(index);
		

		// find node with this index
		ParseNode node;
		for (int j = 0; j < parse_nodes.size(); j++)
		{
			if (parse_nodes[j].indexStr == index)
			{
				node = parse_nodes[j];
				break;
			}
		}

		stack.push(node.right);
		stack.push(node.left);

		i++;
	}

	for (int i = 0; i < leavesIndexes.size(); i++)
	{
		indexes.push_back(leavesIndexes[i]);
	}

	nodes = vector<Node>();

	for (int i = 0; i < indexes.size(); i++)
	{
		Node node;
		int left = -1;
		int right = -1;

		if (indexes[i][0] == 't')
		{
			ParseNode parseNode;
			for (int j = 0; j < parse_nodes.size(); j++)
			{
				if (parse_nodes[j].indexStr == indexes[i])
				{
					parseNode = parse_nodes[j];
					break;
				}
			}

			for (int j = i + 1; j < indexes.size(); j++)
			{
				if (parseNode.left == indexes[j])
				{
					left = j;
				}
				if (parseNode.right == indexes[j])
				{
					right = j;
				}
			}



			node = Node{ left, right, -1, -1, nullptr, nullptr,nullptr, parseNode.operation[0] };

		}
		else
		{
			int shapeIdx = indexes[i][0] == 's' ? 1 : 2;
			node = Node{ -1, -1, -1, shapeIdx, nullptr, nullptr,nullptr, 0 };
		}
		nodes.push_back(node);
	}

	// add parents
	for (int i = 0; i < nodes.size(); i++)
	{
		if (nodes[i].left != -1)
		{
			nodes[nodes[i].left].parent = i;
			nodes[nodes[i].right].parent = i;
		}
	}

	return true;
}

void TreeParser::AttachShapes(Cube* dev_cubes, Sphere* dev_spheres, Cylinder* dev_cylinders)
{
	for (int i = 0; i < leavesIndexes.size(); i++)
	{
		if (leavesIndexes[i][0] == 's')
		{
			nodes[i + num_spheres + num_cubes +num_cylinders- 1].sphere = &dev_spheres[stoi(leavesIndexes[i].substr(1))];
			nodes[i + num_spheres + num_cubes + num_cylinders - 1].shape = 1;
		}
		else if (leavesIndexes[i][0] == 'w')
		{
			nodes[i + num_spheres + num_cubes + num_cylinders - 1].cylinder = &dev_cylinders[stoi(leavesIndexes[i].substr(1))];
			nodes[i + num_spheres + num_cubes + num_cylinders - 1].shape = 3;
		}
		else
		{
			nodes[i + num_spheres + num_cubes + num_cylinders - 1].cube = &dev_cubes[stoi(leavesIndexes[i].substr(1))];
			nodes[i + num_spheres + num_cubes + num_cylinders - 1].shape = 2;
		}
	}
}