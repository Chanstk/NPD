#include"Parameter.h"

Parameter::Parameter(const float& dr,
                     const float& far,
                     const int&tl,
                     const int& stages,
                     const int& size,
                     const int& minLeaf):DR(dr),
                    FAR(far), tree_level(tl),
                    max_stage(stages), obj_size(size),
                    minLeaf(minLeaf)
{
	return ;
}

