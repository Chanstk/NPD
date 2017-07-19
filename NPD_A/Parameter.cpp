#include"Parameter.h"

void Parameter::paraDefine(){
    MINDR =  1;
    MAXFAR = 1e-5;
    tree_level = 6;
    max_stage = 1000;
    obj_size = 20;
    finalNegs = 2000;
    minSamples = 2000;
    negRatio = 2;
    minNegRatio = 0.222;
    trimFrac = 0.05;		// weight trimming in AdaBoost
    samFrac = 1.0;			// the fraction of samples randomly selected in each iteration
    // for training; could be used to avoid overfitting.
    minLeafFrac = 0.01;		// minimal sample fraction w.r.t.the total number of
    // samples required in each leaf node.This is used to avoid overfitting.
    minLeaf = 100;			// minimal samples required in each leaf node.This is used to avoid overfitting.
    maxWeight = 100;		// maximal sample weight in AdaBoost; used to ensure numerical stability.
    numThreads = 10;		// the number of computing threads in tree learning
    numPosSample = 30000;
    modelName = "Tree.xml";
	bootNum = 10000;
}

