#include"Parameter.h"

void Parameter::paraDefine(){
    MINDR =  1;
    MAXFAR = 1e-3;
    tree_level = 8;
    max_stage = 200;
    obj_size = 20;
    finalNegs = 10;
    minSamples = 10;
    negRatio = 1;
    windSize = 20;
    minNegRatio = 1;
    trimFrac = 0.05;		// weight trimming in AdaBoost
    samFrac = 1.0;			// the fraction of samples randomly selected in each iteration
    // for training; could be used to avoid overfitting.
    minLeafFrac = 0.01;		// minimal sample fraction w.r.t.the total number of
    // samples required in each leaf node.This is used to avoid overfitting.
    minLeaf = 10;			// minimal samples required in each leaf node.This is used to avoid overfitting.
    maxWeight = 100;		// maximal sample weight in AdaBoost; used to ensure numerical stability.
    numThreads = 10;		// the number of computing threads in tree learning
    numPosSample = 50;
    bootNum = 1000;
}
