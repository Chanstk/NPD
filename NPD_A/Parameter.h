#ifndef PARAMETER_H
#include <iostream>
#include "pugixml.hpp"
#define PARAMETER_H

class Parameter {
	public:
	float MINDR;
	float MAXFAR;
	int tree_level;
	int max_stage;
	int obj_size;
    int finalNegs;
    int minSamples;
    float negRatio;
    char* modelName;
    // minimal fraction of negative samples required to remain,
    
    // w.r.t.the total number of negative samples.This is a signal of
    // requiring new negative sample bootstrapping.Also used to avoid
    // overfitting.
    double	minNegRatio;
    int bootNum;
    int numPosSample;
    double	trimFrac;		// weight trimming in AdaBoost
    double	samFrac;			// the fraction of samples randomly selected in each iteration
    // for training; could be used to avoid overfitting.
    double	minLeafFrac;		// minimal sample fraction w.r.t.the total number of
    // samples required in each leaf node.This is used to avoid overfitting.
    int	minLeaf;			// minimal samples required in each leaf node.This is used to avoid overfitting.
    double	maxWeight ;		// maximal sample weight in AdaBoost; used to ensure numerical stability.
    int		numThreads;		// the number of computing threads in tree learning
    Parameter(){};
    void paraDefine();
};
#endif
