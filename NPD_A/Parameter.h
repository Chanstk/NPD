#ifndef PARAMETER_H
#define PARAMETER_H

class Parameter {
	public:
	float DR;
	float FAR;
	int tree_level;
	int max_stage;
	int obj_size;
    int minLeaf;
	Parameter(const float&, const float&, const int&, const int&, const int&, const int &);
};
#endif
